import torch as th
from components.episode_buffer import EpisodeBatch
from .q_plus_learner import QPlusLearner


# based on q_learner
class ODITSLearner(QPlusLearner):
    ''' Altering features of q_plus_learner:
        1. using agent's output "mixer_input" instead of "state" as mixer's input
    '''

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

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
        mixer_input = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, train_mode=True, require_logs=require_logs, **kwargs)

            mac_out.append(agent_outs["q"])
            mixer_input.append(agent_outs["mixer_input"])

            if require_logs and 'logs' in agent_outs:
                logs.append(agent_outs['logs'])
            if 'losses' in agent_outs:
                losses.append(agent_outs['losses'])

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mixer_input = th.stack(mixer_input[:-1], dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mixer_input = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, **kwargs)
            target_mac_out.append(target_agent_outs["q"])
            target_mixer_input.append(agent_outs["mixer_input"])

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_mixer_input = th.stack(target_mixer_input[1:], dim=1)

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
            # ===== Mask mixer_input as batch =====
            mixer_input = {"state": mixer_input}
            target_mixer_input = {"state": target_mixer_input}

            chosen_action_qvals = self.mixer(chosen_action_qvals, mixer_input)
            target_max_qvals = self.target_mixer(target_max_qvals, target_mixer_input)

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
