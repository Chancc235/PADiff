import os
import pickle

import numpy as np
import torch as th
from components.episode_buffer import MetaReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from meta.individual import Individual
from modules.encoders import REGISTRY as en_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.config_utils import update_args
from utils.logging import Logger, get_logger


class Stage3Individual(Individual):

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
        self.explore_runner = r_REGISTRY["meta"](self.args, None, pp, prefix="explore")
        self.runner = r_REGISTRY[self.args.runner](self.args, self.logger, pp, prefix="exploit")

        # Set up schemes and groups here
        self.alg2agent = {}
        self.alg2agent["explore"] = self.alg2agent["target"] = self.args.alg2agent["controllable"]
        self.alg2agent["teammate"] = self.args.alg2agent["teammate"]
        self.alg_set = self.alg2agent.keys()
        self.args.agent_ids = self.alg2agent["target"]

        env_info = self.runner.get_env_info()
        self.args.env_info = env_info
        self.args.n_env_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]
        self.args.state_dim = int(np.prod(self.args.state_shape))
        self.args.n_agents = len(self.args.agent_ids)
        self.args.n_ally_agents = self.args.n_env_agents - self.args.n_agents

        # Default/Base scheme
        self.scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},

            "task_embeddings": {"vshape": (self.args.z_dim,), "episode_const": True}
        }
        self.preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)]),
        }

        # Define ReplayBuffer
        self.global_groups = {
            "agents": self.args.n_env_agents
        }
        self.buffer = MetaReplayBuffer(self.scheme, self.global_groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                       preprocess=self.preprocess, device="cpu" if self.args.buffer_cpu_only else self.args.device)

        # setup runner for this experinment
        self.explore_runner.setup(self.scheme, self.global_groups, self.preprocess, self)
        self.runner.setup(self.scheme, self.global_groups, self.preprocess, self)

        self.groups = {
            "agents": self.args.n_agents
        }

        # Setup target agent controller here
        explore_args = update_args(self.args, self.args.explore_alg)
        self.explore_mac = mac_REGISTRY["partial"](self.buffer.scheme, self.groups,  explore_args)
        explore_load_path = os.path.join(self.args.local_saves_path, self.args.explore_load_path)
        max_ts = str(max(map(int, os.listdir(explore_load_path))))
        explore_load_path = os.path.join(explore_load_path, max_ts)
        self.explore_mac.load_models(explore_load_path)
        self.encoder = en_REGISTRY[explore_args.encoder](
            explore_args.state_dim, explore_args.rnn_hidden_dim, explore_args.z_dim, explore_args)
        self.encoder.load_state_dict(th.load(f"{explore_load_path}/encoder.th", map_location=lambda storage, loc: storage))

        self.explore_mode = False

        self.mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, self.groups,  self.args)
        self.args.obs_dim = self.mac.input_shape
        self.alg2mac = {"explore": self.explore_mac, "target": self.mac}

        # Learner
        self.learner = le_REGISTRY[self.args.learner](self.mac, self.buffer.scheme, self.logger, self.args)
        if self.args.use_cuda:
            self.explore_mac.cuda()
            self.encoder.cuda()
            self.learner.cuda()

        self.first_train = True
        self.first_set = True

    def train(self):
        """ train the target agent """
        done = False
        if self.first_train:
            # for kmeans agent
            if getattr(self.args, "kmeans", False):
                file_name = "results/teammate_embeddings.pkl"
                if os.path.exists(file_name):
                    X, _, _ = pickle.load(open(file_name, "rb"))
                else:
                    X = []
                    for i, teammate in enumerate(self.pop.individuals):
                        print(i, teammate)
                        self.pop.load_specific_agents(i)
                        for _ in range(self.args.points_per_teammate // self.args.batch_size_run):
                            z = self.model_teammate().cpu().numpy()
                            X.append(z)
                    # for i, teammate in enumerate(self.pop.test_individuals):
                    #     print(i, teammate)
                    #     self.pop.load_specific_agents(i, mode='test')
                    #     for _ in range(self.args.points_per_teammate // self.args.batch_size_run):
                    #         z = self.model_teammate().cpu().numpy()
                    #         X.append(z)
                    X = np.concatenate(X, axis=0)
                self.mac.agent.set_datatset(X)
                self.learner.target_mac.agent.kmeans = self.mac.agent.kmeans
            self._initialize_training_time()
        if self.runner.t_env > self.args.t_max:
            self._test_and_log()
            done = True
            self.logger.console_logger.info("[BRI] Reach t_max, stop training")
            self.runner.close_env()
        else:
            n_train_runs = self.args.episodes_per_teammate // self.runner.batch_size
            for i in range(n_train_runs):
                # first episode, run with explore_policy and model teammates
                z = self.model_teammate()

                # second episode, run with utilize_policy with "z" as extra input
                episode_batch = self.runner.run(test_mode=False,
                                                status_recorder=self.status,
                                                log_train_status=(i == n_train_runs-1),
                                                task_embeddings=z
                                                )
                self.buffer.insert_episode_batch(episode_batch)

                # ! Keep Sample-Train Balance, add this loop to original 'pymarl'
                for _ in range(self.runner.batch_size):
                    if self.buffer.can_sample(self.args.batch_size):
                        episode_sample = self.buffer.sample(self.args.batch_size)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != self.args.device:
                            episode_sample.to(self.args.device)

                        local_batch = self.buffer.select(episode_sample, self.alg2agent["target"])
                        self.learner.train(local_batch, self.runner.t_env, self.episode, global_batch=episode_sample)

                # Execute test runs once in a while
                if (self.runner.t_env - self.last_test_T) / self.args.test_interval >= 1.0:
                    self._test_and_log()

                self.episode += self.args.batch_size_run

                if (self.runner.t_env - self.last_log_T) >= self.args.log_interval:
                    self.logger.log_stat("episode", self.episode, self.runner.t_env)
                    self.last_log_T = self.runner.t_env

        return done

    def model_teammate(self):
        self.explore_mode = True    # use explore_mode to control which mac to use when forward
        with th.no_grad():
            batch = self.explore_runner.run(test_mode=True)
            bs = batch.batch_size
            bl = batch.max_seq_length

            states = batch["state"][:, :-1]
            h = self.encoder.init_hidden(bs)
            for t in range(bl-1):
                z, h = self.encoder(states[:, t], h, test_mode=True)

        self.explore_mode = False
        return z.clone().detach()

    def test(self):
        ''' do testing with all individuals in pop '''
        n_test_runs = max(1, self.args.test_nepisode // self.args.batch_size_run)
        for teammate_id, teammate in enumerate(self.pop.test_individuals):
            self.pop.load_specific_agents(teammate_id, mode='test')
            for _ in range(n_test_runs):
                # load specific agent models
                z = self.model_teammate()

                self.runner.run(test_mode=True,
                                task_embeddings=z,
                                status_recorder=self.status,
                                n_test_episodes=n_test_runs * self.args.batch_size_run * self.pop.n_test_individuals,
                                )

    # -------------------- mask individual as a mac for usage of runner --------------------
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, **kwargs):
        # Only select actions for the selected batch elements in bs
        dim0 = len(bs) if bs != slice(None) else 1
        chosen_actions = th.zeros([dim0, self.args.n_env_agents], dtype=th.long).to(ep_batch.device)
        for alg in self.alg_set:
            # two modes option
            if (self.explore_mode and alg == "target") or (not self.explore_mode and alg == "explore"):
                continue
            if len(self.alg2agent[alg]) > 0:
                true_test_mode = test_mode or alg != "target"
                selected_batch = self.buffer.select(ep_batch, self.alg2agent[alg])
                agent_actions = self.alg2mac[alg].select_actions(
                    selected_batch, t_ep, t_env, bs, test_mode=true_test_mode, global_batch=ep_batch, **kwargs)
                chosen_actions[:, self.alg2agent[alg]] = agent_actions.to(ep_batch.device)

        return chosen_actions
