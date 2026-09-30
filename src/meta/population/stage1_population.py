import os
import time

import numpy as np
import torch as th
from meta.individual import REGISTRY as ind_REGISTRY
from meta.population import Population
from torch.optim import RMSprop
from utils.config_utils import update_args
from utils.timehelper import time_str


class Stage1Population(Population):
    def __init__(self, args, global_logger) -> None:
        super().__init__(args, global_logger)
        self.args = update_args(self.args, self.args.population_alg)

        # configure extra tensorboard logger for population level information
        if args.use_tensorboard:
            tb_root = os.path.join(args.local_results_path, args.unique_token,  'tb_logs')
            global_logger.setup_tb(tb_root)

        self.individuals = self._init_individuals(self.args.population_size)

        # params: all individuals' Q-networks' parameter
        self.params = []
        for ind_id in range(self.pop_size):
            self.params += list(self.individuals[ind_id].mac.parameters())

        self.meta_optimizer = RMSprop(params=self.params, lr=self.args.lr,
                                      alpha=self.args.optim_alpha, eps=self.args.optim_eps)

        self.meta_epoch = 0   # times of calling train_meta
        self.episodes = 0   # num of episodes

        self.status = {
            'meta_loss': 0,
            'test_return_mean': 0   # mean of all individuals
        }
        self.dones = [False for _ in self.individuals]

    def _init_individuals(self, num):
        return [ind_REGISTRY[self.args.ind](id, self.args, self.pp) for id in range(num)]

    def _get_diversity_loss(self):
        '''calculate diversity loss among all individuals'''
        # gather local current trajectories of all individual
        full_tra, masks = [], []
        for ind_id in range(self.pop_size):
            # cur_batch = self.individuals[ind_id].fetch_newest_batch(self.args.inner_runs * self.args.batch_size_run)
            cur_batch = self.individuals[ind_id].fetch_newest_batch(
                self.args.cur_episodes)
            full_tra.append(cur_batch)
            # compute mask
            terminated = cur_batch["terminated"][:, :-1].float()
            mask = cur_batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            masks.append(mask)
        # shape = [pop_size, inner_runs*batch_size_run, trajectory_len, 1]
        full_mask = th.stack(masks, dim=0)
        pi_i_tau = [self.individuals[ind_id].full_compute_prob(
            full_tra) for ind_id in range(self.pop_size)]
        # shape = [pop_size, pop_size, inner_runs*batch_size_run, trajectory_len, n_agents]
        pi_i_tau = th.stack(pi_i_tau, dim=0)
        # shape = [pop_size, inner_runs*batch_size_run, trajectory_len, n_agents]
        pi_hat_tau = th.mean(pi_i_tau, dim=0)

        # difference loss
        abs_loss = th.abs(pi_i_tau - pi_hat_tau) * full_mask
        abs_loss = abs_loss.mean()

        # maximize difference
        return -abs_loss

    def train_inner(self, ind_id: int):
        '''inner loop for an individual'''
        self.dones[ind_id] = self.individuals[ind_id].train()

    def train_inner_all(self):
        '''inner loop for whole population'''
        for ind_id in range(self.pop_size):
            if not self.dones[ind_id] or self.args.optimize_meta:
                self.train_inner(ind_id)

        # update status
        if "test_battle_won_mean" in self.individuals[0].status:
            test_battle_wons = [
                self.individuals[i].status['test_battle_won_mean'] for i in range(self.pop_size)]
            self.status['test_battle_won_mean'] = np.mean(test_battle_wons)
        if "test_return_mean" in self.individuals[0].status:
            test_returns = [
                self.individuals[i].status['test_return_mean'] for i in range(self.pop_size)]
            self.status["test_return_mean"] = np.mean(test_returns)

    def train_meta(self):
        '''maximize diversity'''
        self.logger.console_logger.info(f"[Population] Meta Train for {self.args.meta_update_times} updates")
        meta_loss_list = []
        for _ in range(self.args.meta_update_times):
            meta_loss = self._get_diversity_loss()
            meta_loss_list.append(meta_loss.item())
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

        meta_t_env = int(max(self.individuals[i].runner.t_env for i in range(self.pop_size)))

        self.logger.log_stat('meta_loss', np.mean(meta_loss_list), meta_t_env)
        self.status['meta_loss'] = np.mean(meta_loss_list)

    def save_population(self):
        # save individuals' current models
        for ind_id in range(self.pop_size):
            self.individuals[ind_id].save_individual()

        # save meta objects & status
        save_root = os.path.join(self.args.local_results_path,
                                 self.args.unique_token, 'meta', str(self.meta_epoch))
        os.makedirs(save_root, exist_ok=True)
        th.save(self.meta_optimizer.state_dict(), f"{save_root}/opt.th")

        # save status
        with open(f'{save_root}/info.txt', 'w') as f:
            # track each individuals' current timestep, in order to load them from correct directory
            f.write('individiaul_timesteps: ')
            for ind_id in range(self.pop_size):
                ind = self.individuals[ind_id]
                f.write(f'{int(ind.runner.t_env)} ')
            f.write('\n')

            # other status
            for k, v in self.status.items():
                f.write(f'{k} : {str(v)}' + '\n')

    def load_population(self, unique_token, meta_epoch):
        save_root = os.path.join(self.args.local_results_path, unique_token)

        # load meta objects
        self.meta_optimizer.load_state_dict(
            th.load(f"{save_root}/meta/{meta_epoch}/opt.th", map_location=lambda storage, loc: storage))

        # load individuals
        with open(f'{save_root}/meta/{meta_epoch}/info.txt', 'r') as f:
            ind_steps = f.readlines()[0].strip().split(' ')[1:]
            assert len(ind_steps) == self.pop_size

            for ind_id in range(self.pop_size):
                self.individuals[ind_id].load_individual(
                    unique_token, ind_id, ind_steps[ind_id])

    def run(self):
        global_start_time = time.time()
        last_save = 0
        while not np.all(self.dones):
            self.logger.console_logger.info(f'================ MetaEpoch: {self.meta_epoch} ================')
            self.logger.console_logger.info(f"Time passed: {time_str(time.time() - global_start_time)}")

            # inner loop
            self.train_inner_all()
            # meta update
            if self.args.optimize_meta:
                self.train_meta()

            self.meta_epoch += 1
            self.episodes += max(1, self.args.inner_loop_episodes // self.args.batch_size_run) * self.args.batch_size_run
            if self.args.save_population and ((self.episodes - last_save >= self.args.save_population_episodes) or np.all(self.dones) or (last_save == 0)):
                self.save_population()
                last_save = self.episodes

        for ind in self.individuals:
            ind.close_env()
