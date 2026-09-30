import os

import numpy as np
import torch as th
from components.episode_buffer import MetaReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from meta.individual import Individual
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger, get_logger


class Stage2Individual(Individual):

    def __init__(self, args, pp, pop):
        super().__init__(args)

        self.pop = pop
        self.args.n_tasks = self.pop.n_individuals
        self.status = {
            'battle_won_mean': 0,
            'test_return_mean': 0,
        }    # track training stage, for saving information while saving model

        # setup individual logger
        self.logger = Logger(get_logger())
        if self.args.use_tensorboard:
            tb_logs_path = os.path.join(self.args.local_results_path, self.args.unique_token, 'tb_logs')
            self.logger.setup_tb(tb_logs_path)

        # Init runner which cantians env info
        self.runner = r_REGISTRY[self.args.runner](self.args, self.logger, pp)

        # Set up schemes and groups here
        self.alg2agent = {}
        self.alg2agent["explore"] = self.args.alg2agent["controllable"]
        self.alg2agent["teammate"] = self.args.alg2agent["teammate"]
        self.alg_set = self.alg2agent.keys()
        self.args.agent_ids = self.alg2agent["explore"]

        # get env information
        env_info = self.runner.get_env_info()
        self.args.env_info = env_info
        self.args.n_env_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]
        self.args.state_dim = int(np.prod(self.args.state_shape))
        self.args.n_agents = len(self.args.agent_ids)
        self.args.n_ally_agents = self.args.n_env_agents - self.args.n_agents
        self.args.ally_ids = [i for i in range(self.args.n_env_agents) if i not in self.args.agent_ids]

        # Default/Base scheme
        self.scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        self.preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)]),
        }

        # Define ReplayBuffer
        self.global_groups = {
            "agents": self.args.n_env_agents
        }
        self.buffer = MetaReplayBuffer(self.scheme, self.global_groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                       preprocess=self.preprocess,
                                       device="cpu" if self.args.buffer_cpu_only else self.args.device)

        # setup runner for this experinment
        self.runner.setup(self.scheme, self.global_groups, self.preprocess, self)

        groups = {
            "agents": self.args.n_agents
        }
        # Setup explore agent controller here
        self.mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, groups,  self.args)
        self.args.obs_dim = self.mac.input_shape
        self.alg2mac = {"explore": self.mac}

        # Learner
        self.learner = le_REGISTRY[self.args.learner](self.mac, self.buffer.scheme, self.logger, self.args)
        if self.args.use_cuda:
            self.learner.cuda()

        self.first_train = True
        self.first_set = True

    def train(self):
        """ train the explore agent """
        done = False
        if self.first_train:
            self._initialize_training_time()
        if self.runner.t_env > self.args.t_max:
            self._test_and_log()
            done = True
            self.logger.console_logger.info("[BRI] Reach t_max, stop training")
            self.runner.close_env()
        else:
            n_train_runs = self.args.episodes_per_teammate // self.runner.batch_size
            for i in range(n_train_runs):
                episode_batch = self.runner.run(test_mode=False,
                                                status_recorder=self.status,
                                                log_train_status=(i == n_train_runs-1)
                                                )
                self.buffer.insert_episode_batch(episode_batch)

                # ! Keep Sample-Train Balance, add this loop to original 'pymarl'
                for j in range(self.runner.batch_size):
                    if self.buffer.can_sample(self.args.batch_size):
                        episode_sample = self.buffer.sample(self.args.batch_size)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != self.args.device:
                            episode_sample.to(self.args.device)

                        local_batch = self.buffer.select(episode_sample, self.alg2agent["explore"])
                        self.learner.train(local_batch, self.runner.t_env, self.episode,
                                           global_batch=episode_sample,
                                           write_log=self.episode % 80 == 0 and j == 0)

                # Execute test runs once in a while
                if (self.runner.t_env - self.last_test_T) / self.args.test_interval >= 1.0:
                    self._test_and_log()

                self.episode += self.args.batch_size_run

                if (self.runner.t_env - self.last_log_T) >= self.args.log_interval:
                    self.logger.log_stat("episode", self.episode, self.runner.t_env)
                    self.last_log_T = self.runner.t_env

        return done

    def test(self):
        ''' do testing with all individuals in pop '''
        n_test_runs = max(1, self.args.test_nepisode // self.runner.batch_size)
        for teammate_id, teammate in enumerate(self.pop.test_individuals):
            # load specific agent models
            self.pop.load_specific_agents(teammate_id, mode='test')
            for _ in range(n_test_runs):
                self.runner.run(test_mode=True,
                                status_recorder=self.status,
                                n_test_episodes=n_test_runs * self.args.batch_size_run * self.pop.n_test_individuals,
                                )

    # -------------------- mask individual as a mac for usage of runner --------------------
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, **kwargs):
        # Only select actions for the selected batch elements in bs
        dim0 = len(bs) if bs != slice(None) else 1
        chosen_actions = th.zeros([dim0, self.args.n_env_agents], dtype=th.long).to(ep_batch.device)
        for alg in self.alg_set:
            if len(self.alg2agent[alg]) > 0:
                true_test_mode = test_mode or alg != "explore"
                selected_batch = self.buffer.select(ep_batch, self.alg2agent[alg])
                agent_actions = self.alg2mac[alg].select_actions(
                    selected_batch, t_ep, t_env, bs, test_mode=true_test_mode, global_batch=ep_batch, **kwargs)
                chosen_actions[:, self.alg2agent[alg]] = agent_actions.to(ep_batch.device)

        return chosen_actions
