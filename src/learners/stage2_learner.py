import copy
import os

import torch as th
from components.episode_buffer import EpisodeBatch
from modules.decoders.mlp_decoder import MLPDecoder
from modules.encoders import REGISTRY as en_REGISTRY
from modules.mixers import REGISTRY as mixer_REGISTRY
from torch.optim import RMSprop, Adam


def to_str(tensor):
    return str(tensor.clone().detach().cpu().numpy())+'\n'


def write_summary(z, teammate_actions, teammate_actions_pred):
    os.makedirs("logs", exist_ok=True)
    with open('logs/tmp.txt', 'a+') as f:
        f.write(to_str(z[0]))
        f.write(to_str(teammate_actions[5]))
        f.write(to_str(teammate_actions_pred[5]))
        f.write(to_str(teammate_actions[-5]))
        f.write(to_str(teammate_actions_pred[-5]))
        f.write('--------------------------------------------------\n')


# based on q_learner
class Stage2Learner:
    ''' adding features to q_learner
        1. Optimize extra "explore_policy" and "encoder" based on reconstruction error
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

        self.encoder = en_REGISTRY[self.args.encoder](args.state_dim, args.rnn_hidden_dim, args.z_dim, args)
        self.decoder = MLPDecoder(args.z_dim, args.state_dim, args.mlp_hidden_dim, args.n_actions * args.n_ally_agents, args)
        # self.params += list(self.encoder.parameters())
        # self.params += list(self.decoder.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.ed_params = list(self.encoder.parameters())+list(self.decoder.parameters())
        self.ed_optimiser = Adam(params=self.ed_params, lr=3e-4)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, write_log: False, **kwargs):
        global_batch = kwargs["global_batch"]
        # Get the relevant quantities
        rewards = batch["reward"][:, : -1]
        actions = batch["actions"][:, : -1]
        terminated = batch["terminated"][:, : -1].float()
        mask = batch["filled"][:, : -1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, : -1])
        avail_actions = batch["avail_actions"]

        # ! adding intrinsic reward
        bs = global_batch.batch_size
        bl = global_batch.max_seq_length
        states = global_batch["state"][:, : -1]
        teammate_actions = global_batch["actions"][:, : -1, self.args.ally_ids].flatten()
        h = self.encoder.init_hidden(bs)
        for t in range(bl-1):
            z, h = self.encoder(states[:, t], h)

        teammate_actions_pred = self.decoder(bl-1, z, states).reshape(-1, self.args.n_actions)
        if write_log:
            write_summary(z, teammate_actions, teammate_actions_pred)

        CE = th.nn.CrossEntropyLoss(reduction='none')
        recon_error = CE(teammate_actions_pred, teammate_actions).reshape(
            bs, bl-1, self.args.n_ally_agents).mean(-1).unsqueeze(-1)
        recon_mask = teammate_actions.reshape(bs, bl-1, self.args.n_ally_agents).sum(-1) > 0
        recon_mask = recon_mask.unsqueeze(-1)
        recon_error = recon_error*recon_mask
        rewards += self.args.intrinsic_alpha * recon_error.clone().detach()

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs["q"])

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, : -1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
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
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch[:, : -1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch[:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        rl_loss = (masked_td_error ** 2).sum() / mask.sum()
        encoder_loss = recon_error.sum() / (mask*recon_mask).sum()

        # Optimise
        self.optimiser.zero_grad()
        rl_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.ed_optimiser.zero_grad()
        encoder_loss.backward()
        ed_grad_norm = th.nn.utils.clip_grad_norm_(self.ed_params, self.args.grad_norm_clip)
        self.ed_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("encoder_loss", encoder_loss.item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm + ed_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() /
                                 (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env

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
        self.encoder.cuda()
        self.decoder.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.encoder.state_dict(), "{}/encoder.th".format(path))
        th.save(self.decoder.state_dict(), "{}/decoder.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict(th.load("{}/encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(th.load("{}/decoder.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
