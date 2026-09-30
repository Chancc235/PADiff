import torch as th

from .partial_controller import PartialMAC

''' adding additional "context" information in agent inputs '''


class ContextMAC_v1(PartialMAC):
    ''' directly read "task_embeddings" [z],
        no extra calculation here
    '''

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_inputs(self, batch, t, **kwargs):
        # sourcery skip: inline-immediately-returned-variable
        bs = batch.batch_size

        # ====== 1. build agent inputs ======
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            raw_obs_agent_ids = th.eye(self.n_env_agents, device=batch.device)
            obs_agent_ids = raw_obs_agent_ids[self.agent_ids]
            inputs.append(obs_agent_ids.unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)

        # ====== 2. build context ======
        z = batch["task_embeddings"]

        input_dict = {
            "obs": inputs,
            "z": z
        }

        return input_dict


class ContextMAC_v2(PartialMAC):
    ''' calculate additional "teammate agents' current step infomation: [o_t^{-1}, a_t^{-1}]
    '''

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_inputs(self, batch, t, **kwargs):
        # sourcery skip: inline-immediately-returned-variable
        global_batch = kwargs["global_batch"]  # assert exist
        bs = global_batch.batch_size

        # ====== 1. build agent inputs ======
        inputs = [global_batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(global_batch["actions_onehot"][:, t]))
            else:
                inputs.append(global_batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_env_agents, device=global_batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_env_agents, -1) for x in inputs], dim=-1)
        self_obs = inputs[:, self.agent_ids].reshape(bs*self.n_agents, -1)

        # ====== 2. build context ======
        teammate_obs = inputs[:, self.ally_ids].reshape(bs, self.n_ally_agents, -1)
        teammate_actions = global_batch["actions"][:, t, self.ally_ids]

        input_dict = {
            "obs": self_obs,
            "teammate_obs": teammate_obs,
            "teammate_actions": teammate_actions,
        }

        return input_dict


class ContextMAC_v3(PartialMAC):
    ''' calculate additional "teammate agents' current step and global state information: [o_t^{-1}, a_t^{-1}, s_t]
    '''

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_inputs(self, batch, t, **kwargs):
        # sourcery skip: inline-immediately-returned-variable
        global_batch = kwargs["global_batch"]  # assert exist
        bs = global_batch.batch_size

        # ====== 1. build agent inputs ======
        inputs = [global_batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(global_batch["actions_onehot"][:, t]))
            else:
                inputs.append(global_batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_env_agents, device=global_batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_env_agents, -1) for x in inputs], dim=-1)
        self_obs = inputs[:, self.agent_ids].reshape(bs*self.n_agents, -1)

        # ====== 2. build context ======
        teammate_obs = inputs[:, self.ally_ids].reshape(bs, self.n_ally_agents, -1)
        teammate_actions = global_batch["actions"][:, t, self.ally_ids]

        # ====== 3. get state for full information ======
        state = global_batch["state"][:, t]

        input_dict = {
            "obs": self_obs,
            "teammate_obs": teammate_obs,
            "teammate_actions": teammate_actions,
            "state": state
        }

        return input_dict


class ContextMAC_v4(PartialMAC):
    ''' directly read "task_embeddings" [z],
        no extra calculation here
    '''

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_inputs(self, batch, t, **kwargs):
        # sourcery skip: inline-immediately-returned-variable
        global_batch = kwargs["global_batch"]  # assert exist
        bs = batch.batch_size

        # ====== 1. build agent inputs ======
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            raw_obs_agent_ids = th.eye(self.n_env_agents, device=batch.device)
            obs_agent_ids = raw_obs_agent_ids[self.agent_ids]
            inputs.append(obs_agent_ids.unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)

        # ====== 2. build context ======
        state = global_batch["state"][:, t]

        input_dict = {
            "obs": inputs,
            "state": state
        }

        return input_dict
