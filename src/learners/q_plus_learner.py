import copy

import torch as th
from components.episode_buffer import EpisodeBatch
from modules.mixers import REGISTRY as mixer_REGISTRY
from torch.optim import RMSprop


# based on q_learner
class QPlusLearner:
    ''' Adding features to q_learner
        1. agent can calculate additional "losses" and return to controller to add to "total_loss"
        2. agent can return additional "logs" to show on tensorboard
    '''

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = mixer_REGISTRY[args.mixer](args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.supervised_loss = bool(hasattr(args, 'supervised_loss_weight') and args.supervised_loss_weight > 0)
        self.supervised_loss_with_mixer = bool(self.supervised_loss and hasattr(args, 'supervised_loss_with_mixer') and args.supervised_loss_with_mixer)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, **kwargs):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # require_logs = t_env - self.log_stats_t >= self.args.learner_log_interval
        require_logs = False

        logs = []
        losses = []

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, train_mode=True, require_logs=require_logs, **kwargs)

            mac_out.append(agent_outs["q"])

            if require_logs and 'logs' in agent_outs:
                logs.append(agent_outs['logs'])
            if 'losses' in agent_outs:
                losses.append(agent_outs['losses'])

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, **kwargs)
            target_mac_out.append(target_agent_outs["q"])

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses)
        loss += external_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() /
                                 (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self._log_for_scalar_and_histogram(logs, t_env)
            self._log_for_loss(loss_dict, t_env)

            self.log_stats_t = t_env

    def _log_for_scalar_and_histogram(self, logs, t):
        if len(logs) == 0:
            return
        keys = list(logs[0].keys())
        for k in keys:
            log_key = '_'.join(k.split('_')[1:])
            if str(k).startswith('Histogram'):
                value = th.stack([l[k] for l in logs], dim=1)
                self.logger.log_histogram(log_key, value, t)
            elif str(k).startswith('Scalar'):
                value = th.tensor([l[k] for l in logs]).mean()
                self.logger.log_stat(log_key, value, t)

    def _process_loss(self, losses: list):
        loss_dict = {}
        loss_num = {}
        for item in losses:
            for k, v in item.items():
                loss_dict[k] = loss_dict.get(k, 0) + v
                # filter zero value
                loss_num[k] = loss_num.get(k, 1)  # get(k,0) may have 0-division
                if v != 0:
                    loss_num[k] += 1

        total_loss = 0
        for k in loss_dict:
            loss_dict[k] /= loss_num[k]
            total_loss += loss_dict[k]

        return total_loss, loss_dict

    def _log_for_loss(self, losses: dict, t):
        for k, v in losses.items():
            self.logger.log_stat(k, v.item(), t)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
