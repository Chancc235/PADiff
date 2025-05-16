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
import pickle
import threading


class CollectorIndividual(Individual):

    def __init__(self, args, pp, pop):
        super().__init__(args)

        self.pop = pop
        self.args.n_tasks = self.pop.n_individuals
        self.status = {
            'battle_won_mean': 0,
            'test_return_mean': 0,
        }   

        self.logger = Logger(get_logger())
        if self.args.use_tensorboard:
            tb_logs_path = os.path.join(self.args.local_results_path, self.args.unique_token, 'tb_logs')
            self.logger.setup_tb(tb_logs_path)

        self.runner = r_REGISTRY[self.args.runner](self.args, self.logger, pp)

        self.alg2agent = {}
        self.alg2agent["explore"] = self.args.alg2agent["controllable"]
        self.alg2agent["teammate"] = self.args.alg2agent["teammate"]
        self.alg_set = self.alg2agent.keys()
        self.args.agent_ids = self.alg2agent["explore"]

        env_info = self.runner.get_env_info()
        self.args.env_info = env_info
        self.args.n_env_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]
        self.args.state_dim = int(np.prod(self.args.state_shape))
        self.args.n_agents = len(self.args.agent_ids)
        self.args.n_ally_agents = self.args.n_env_agents - self.args.n_agents
        self.args.ally_ids = [i for i in range(self.args.n_env_agents) if i not in self.args.agent_ids]

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

        self.global_groups = {
            "agents": self.args.n_env_agents
        }
        self.buffer = MetaReplayBuffer(self.scheme, self.global_groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                       preprocess=self.preprocess,
                                       device="cpu" if self.args.buffer_cpu_only else self.args.device)

        

        groups = {
            "agents": self.args.n_agents
        }
        self.mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, groups, self.args)
        self.args.obs_dim = self.mac.input_shape
        self.alg2mac = {"explore": self.mac}

        self.runner.setup(self.scheme, self.global_groups, self.preprocess, self.mac)

        self.first_set = True 
        self.first_train = True
    def init_buffer(self):
        """初始化缓冲区"""
        self.buffer = MetaReplayBuffer(self.scheme, self.global_groups, self.args.buffer_size, self.args.env_info["episode_limit"] + 1,
                                       preprocess=self.preprocess,
                                       device="cpu" if self.args.buffer_cpu_only else self.args.device)
    def collect_trajectories(self):
        """收集轨迹"""
        done = False

        if self.first_train:
            self.first_train = False
            self._initialize_training_time()

        while not done:
            self.logger.console_logger.info(f"Runing batch")

            episode_batch = self.runner.run(test_mode=True, status_recorder=self.status)
            self.logger.console_logger.info(f"Get batch")
            if episode_batch.device != self.args.device:
                episode_batch.to(self.args.device) 

            self.buffer.insert_episode_batch(episode_batch)
            self.logger.console_logger.info(f"episode {self.episode} Inserted")
            self.episode += self.args.batch_size_run

            if self.episode >= self.args.t_max:
                done = True

            if (self.runner.t_env - self.last_log_T) >= self.args.log_interval:
                self.logger.log_stat("episode", self.episode, self.runner.t_env)
                self.last_log_T = self.runner.t_env
                

            return done


    def save_trajectories(self):
        
        buffer = self.buffer.get_all_transitions()
        '''
        buffer_data = self.buffer.fetch_newest_batch(self.args.save_BR_episodes)
        buffer = {
            "transition_data": buffer_data.data.transition_data,
            "episode_data": buffer_data.data.episode_data
        }
        '''

        try:
            pickle.dumps(buffer)
        except Exception as e:
            self.logger.console_logger.error(f"Data not serializable: {e}")
            return
            
        save_path = f"{self.args.local_saves_path}/trajectorys/buffer_{self.episode}.pkl"
        if not os.path.exists(save_path):
            lock = threading.Lock()
            with lock:
                with open(save_path, 'wb') as f:
                    pickle.dump(buffer, f)
                    f.flush()
                    os.fsync(f.fileno())
            self.logger.console_logger.info(f"Trajectories saved to {save_path}.")

    def test(self):

        n_test_runs = max(1, self.args.test_nepisode // self.runner.batch_size)
        for teammate_id, teammate in enumerate(self.pop.test_individuals):
            self.pop.load_specific_agents(teammate_id, mode='test')
            for _ in range(n_test_runs):
                self.runner.run(test_mode=True,
                                status_recorder=self.status,
                                n_test_episodes=n_test_runs * self.args.batch_size_run * self.pop.n_test_individuals)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, **kwargs):
   
        device = self.args.device 
 
        dim0 = len(bs) if bs != slice(None) else 1
        chosen_actions = th.zeros([dim0, self.args.n_env_agents], dtype=th.long).to(device)

        for alg in self.alg_set:
            if len(self.alg2agent[alg]) > 0:
                true_test_mode = test_mode or alg != "explore"


                selected_batch = self.buffer.select(ep_batch, self.alg2agent[alg])
                selected_batch.to(device) 
                agent_actions = self.alg2mac[alg].select_actions(
                    selected_batch, t_ep, t_env, bs, test_mode=true_test_mode, global_batch=ep_batch, **kwargs
                )

                chosen_actions[:, self.alg2agent[alg]] = agent_actions.to(device)

        return chosen_actions

