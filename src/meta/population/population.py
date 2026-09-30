import copy
import os
import random

import numpy as np
from runners.meta_runner import ParallelProcessor
from utils.rl_utils import train_test_split


def _invert(x):
    return 1 / (x + 1e-10)  # incase of 0-division


def _get_prob(x, beta=2):
    x = np.power(x, beta)
    return x / np.sum(x)


class Population:
    def __init__(self, args, global_logger):
        self.args = copy.deepcopy(args)
        self.logger = global_logger

        # share this processor among all individuals
        if self.args.runner not in ["episode", "parallel"]:
            self.pp = ParallelProcessor(self.args)
        else:
            self.pp = None

    @property
    def pop_size(self):
        return len(self.individuals)

    def run(self):
        raise NotImplementedError


class StrPopulation(Population):
    ''' store each individual's model path in self.individuals 
    '''

    def __init__(self, args, global_logger):
        super().__init__(args, global_logger)

        # ==== 1. prepare for indiduals ======
        self.individuals = []   # short for train_individuals
        self.test_individuals = []

        if hasattr(self.args, "population_directories"):
            # support single directory
            if type(self.args.population_directories) is str:
                self.args.population_directories = [self.args.population_directories]
            for pop_dir in self.args.population_directories:
                root_dir = os.path.join(self.args.local_saves_path, pop_dir)
                for ind_name in os.listdir(root_dir):
                    sub_root = os.path.join(root_dir, ind_name)
                    # selected history checkpoints
                    checkpoint_timesteps = sorted(map(int, os.listdir(sub_root)))
                    checkpoint_len = len(checkpoint_timesteps)
                    if self.args.use_history:
                        self.individuals.append(os.path.join(sub_root, str(checkpoint_timesteps[(checkpoint_len)//3])))  # 1/3
                        self.individuals.append(os.path.join(sub_root, str(checkpoint_timesteps[(2*checkpoint_len)//3])))  # 2/3
                    self.individuals.append(os.path.join(sub_root, str(checkpoint_timesteps[-1])))    # final

            if hasattr(self.args, 'test_population_directories'):
                # support single directory
                if type(self.args.test_population_directories) is str:
                    self.args.test_population_directories = [self.args.test_population_directories]
                for pop_dir in self.args.test_population_directories:
                    root_dir = os.path.join(self.args.local_saves_path, pop_dir)
                    for ind_name in os.listdir(root_dir):
                        sub_root = os.path.join(root_dir, ind_name)
                        max_ts = str(max(map(int, os.listdir(sub_root))))
                        self.test_individuals.append(os.path.join(sub_root, max_ts))

            elif self.args.train_test_split:
                self.individuals, self.test_individuals = train_test_split(self.individuals, test_percent=self.args.test_percent)
            else:
                self.test_individuals = self.individuals
            self.print_pop_info()
        else:
            if not hasattr(self.args, "population_composition"):
                self.args.population_composition = []
            self.individuals = best_individuals = self.test_individuals = self.args.population_composition

        random.shuffle(self.individuals)
        random.shuffle(self.test_individuals)
        self.n_individuals = len(self.individuals)
        self.n_test_individuals = len(self.test_individuals)

        self.last_sample = -1  # record last sampled individual id

    def sample_individual(self, sample_mode="static"):
        # sourcery skip: extract-method
        '''sample individuals from population'''
        if sample_mode == "static":
            self.last_sample += 1
            if self.last_sample >= self.n_individuals:
                self.last_sample = 0
            return self.last_sample, str(self.individuals[self.last_sample])
        elif sample_mode == "weighted":
            invert_status = list(map(_invert, self.status))
            argsort = np.argsort(np.array(invert_status))
            invert_rank = np.zeros_like(argsort)
            for i in range(argsort.shape[0]):
                i = int(i)
                invert_rank[argsort[i]] = i + 1  # let rank start from 1 instead of 0
            prob = _get_prob(invert_rank)
            i = np.random.choice(len(self.individuals), p=prob)
            return i, str(self.individuals[i])
        else:
            i = np.random.randint(len(self.individuals))
            return i, str(self.individuals[i])

    def load_specific_agents(self, teammate_id, mode="train"):
        ''' load specific agents denoted by type_num '''
        if mode == "train":
            teammate = self.individuals[teammate_id]
        elif mode == "test":
            teammate = self.test_individuals[teammate_id]
        else:
            raise NotImplementedError

        self.BRI.set_agents(teammate)

    def print_pop_info(self):
        ''' show important information of population '''
        # train_individuals
        print("========> Train Pop <========")
        test_battle_won = []
        for ind_path in self.individuals:
            status_file = os.path.join(ind_path, 'status.txt')
            with open(status_file, 'r') as f:
                line = f.readline()
                test_battle_won.append(float(line.split()[-1]))
        print("Size:", len(self.individuals), "win_mean:", np.mean(test_battle_won))

        # test_individuals
        print("========> Test Pop <=========")
        test_battle_won = []
        for ind_path in self.test_individuals:
            status_file = os.path.join(ind_path, 'status.txt')
            with open(status_file, 'r') as f:
                line = f.readline()
                test_battle_won.append(float(line.split()[-1]))
        print("Size:", len(self.test_individuals), "win_mean:", np.mean(test_battle_won))
