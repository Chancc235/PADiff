import copy

import torch as th
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY

from .basic_controller import BasicMAC

"""
Adding [agent_ids] specification to control "a part of" all agents in the environment
In order to control all agents, each agent group requires one PartialMAC.
"""


class PartialMAC(BasicMAC):
    """ Common [neural network agent] controller
        This multi-agent controller shares parameters between agents within its dominion
    """

    def __init__(self, scheme, groups, args):
        self.args = args
        self.type = "network"

        # a little bit waste of code, but more precise when used for noncontrollable agents.
        self.n_agents = args.n_agents    # num of 'controllable' agents
        self.n_env_agents = args.n_env_agents  # num of all agents / entities
        self.n_ally_agents = self.n_env_agents - self.n_agents
        self.agent_ids = args.agent_ids
        self.ally_ids = [i for i in range(self.n_env_agents) if i not in self.agent_ids]

        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def _build_inputs(self, batch, t, **kwargs):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            raw_obs_agent_ids = th.eye(self.n_env_agents, device=batch.device)
            obs_agent_ids = raw_obs_agent_ids[self.agent_ids]
            inputs.append(obs_agent_ids.unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)
        rets = {"obs": inputs}
        return rets

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_env_agents

        return input_shape


class PartialRuleMAC:
    """ Control [rule-based agents], supporting policy type switching """

    def __init__(self, scheme, group, args):
        self.args = copy.deepcopy(args)
        self.type = "rule"

        self.n_agents = args.n_agents    # num of 'controllable' agents
        assert self.n_agents == 1, "The num of rule-based agent should be one."

        self._build_agents(args.env_info)

    def _build_agents(self, env_info):
        """ Build Rule-based agents. """
        policy_types = self.args.population_composition
        if self.args.env == "overcooked":
            self.agent = agent_REGISTRY["overcooked_rule"](
                terrain_mtx=env_info["terrain_mtx"], policy_type=policy_types[0], env_parallel=self.args.batch_size_run, args=self.args)
        elif self.args.env == "lbf":
            self.agent = agent_REGISTRY["lbf_rule"](args=self.args)
        else:
            raise ValueError(f"Unsupported env {self.args.env}!")

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), dynamic_env_infos=None, **kwargs):
        # Only select actions for the selected batch elements in bs
        chosen_actions = []
        for env_idx in bs:
            state = dynamic_env_infos[env_idx]["state"]
            chosen_action = self.agent.act(state, env_idx, t_ep == 0)
            chosen_actions.append(chosen_action)
        return th.as_tensor(chosen_actions).reshape(-1, 1)

    def switch_policy_type(self, policy_type):
        """ Switch policy type. """
        self.agent.switch_policy_type(policy_type)
