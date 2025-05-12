import os
import time

import torch as th
from components.episode_buffer import MetaReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from meta.individual import Individual
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger, get_logger
from utils.timehelper import time_left, time_str


class Stage1Individual(Individual):
    def __init__(self, individual_id, args, pp=None):
        super().__init__(args)

        self.individual_id = individual_id
        self.name = str(individual_id)
        self.status = {
            'test_return_mean': 0,
        }    # track training stage, for saving information while saving model

        # setup individual logger
        self.logger = Logger(get_logger())
        if self.args.use_tensorboard:
            tb_logs_path = os.path.join(self.args.local_results_path, self.args.unique_token, 'tb_logs', self.name)
            self.logger.setup_tb(tb_logs_path)

        # Init runner which cantians env info
        if pp is not None:
            self.runner = r_REGISTRY[self.args.runner](self.args, self.logger, pp)
        else:
            self.runner = r_REGISTRY[self.args.runner](self.args, self.logger)

        # Set up schemes and groups here
        env_info = self.runner.get_env_info()
        self.args.n_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": self.args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])
        }

        self.buffer = MetaReplayBuffer(scheme, groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                       preprocess=preprocess,
                                       device="cpu" if self.args.buffer_cpu_only else self.args.device)

        # Setup multiagent controller here
        self.mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, groups, self.args)

        # Give runner the scheme
        self.runner.setup(scheme, groups, preprocess, self.mac)

        # Learner
        self.learner = le_REGISTRY[self.args.learner](self.mac, self.buffer.scheme, self.logger, self.args)

        if self.args.use_cuda:
            self.learner.cuda()

        self.first_train = True

    def train(self):  # sourcery skip: extract-method
        # partial train
        if self.first_train:
            self.episode = 0
            self.last_test_T = -self.args.test_interval - 1
            self.last_log_T = 0
            self.model_save_time = 0

            self.start_time = time.time()
            self.last_time = time.time()
            self.first_train = False
        done = self.runner.t_env > self.args.t_max
        self.logger.console_logger.info(f"[Individual {self.name}] Begin training for {self.args.inner_loop_episodes} episodes")
        n_runs = max(1, self.args.inner_loop_episodes // self.runner.batch_size)
        for i in range(n_runs):
            # Parallel Run for 'runner.batch_size' episode at a time
            episode_batch = self.runner.run(test_mode=False,
                                            status_recorder=self.status,
                                            log_train_status=(i == n_runs-1))
            self.buffer.insert_episode_batch(episode_batch)

            # ! Keep Sample-Train Balance, add this loop to original 'pymarl'
            for _1 in range(self.runner.batch_size):
                if self.buffer.can_sample(self.args.batch_size):
                    episode_sample = self.buffer.sample(self.args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != self.args.device:
                        episode_sample.to(self.args.device)

                    self.learner.train(episode_sample, self.runner.t_env, self.episode)

            # Execute test runs once in a while
            n_test_runs = max(1, self.args.test_nepisode // self.runner.batch_size)
            if (self.runner.t_env - self.last_test_T) / self.args.test_interval >= 1.0:

                self.logger.console_logger.info(f"t_env: {self.runner.t_env} / {self.args.t_max}")

                self.logger.console_logger.info(
                    f"Estimated time left: {time_left(self.last_time, self.last_test_T, self.runner.t_env, self.args.t_max)}. Time passed: {time_str(time.time() - self.start_time)}")

                self.last_time = time.time()

                self.last_test_T = self.runner.t_env

                for _ in range(n_test_runs):
                    self.runner.run(test_mode=True, status_recorder=self.status)

            self.episode += self.args.batch_size_run

            if (self.runner.t_env - self.last_log_T) >= self.args.log_interval:
                self.logger.log_stat("episode", self.episode, self.runner.t_env)
                self.last_log_T = self.runner.t_env

        return done

    def fetch_newest_batch(self, batch_size):
        """ fetch newest episode batch for meta gradient optimization """
        nb = self.buffer.fetch_newest_batch(batch_size)
        if nb.device != self.args.device:
            nb.to(self.args.device)
        return nb

    def compute_prob(self, ep_batch):
        """compute chosen-action probability"""
        actions = ep_batch["actions"][:, :-1]

        # Calculate estimated Q-values
        action_probs_out = []
        self.mac.init_hidden(ep_batch.batch_size)
        for t in range(ep_batch.max_seq_length):
            agent_outs = self.mac.forward(ep_batch, t=t)["q"]
            action_probs = th.exp(agent_outs)/th.exp(agent_outs).sum(dim=-1).unsqueeze(-1)
            action_probs_out.append(action_probs)
        action_probs_out = th.stack(action_probs_out, dim=1)

        return th.gather(action_probs_out[:, :-1], dim=3, index=actions).squeeze(3)

    def full_compute_prob(self, ep_batchs):
        """compute trajectory probability for a list of ep_batch (from all individuals)"""
        ret = [self.compute_prob(ep_batch) for ep_batch in ep_batchs]
        ret = th.stack(ret, dim=0)
        return ret
