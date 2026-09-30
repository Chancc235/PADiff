import time

from meta.individual import REGISTRY as ind_REGISTRY
from meta.population import StrPopulation
from utils.config_utils import update_args
from utils.timehelper import time_str
import os

class CollectorPopulation(StrPopulation):
    '''Modified version for trajectory collection.
       Randomly chooses a teammate model, interacts with the environment, and saves the generated trajectories.
    '''

    def __init__(self, args, global_logger) -> None:
        super().__init__(args, global_logger)
        self.args = update_args(self.args, self.args.explore_alg)

        # ====== record collection status ======
        self.status = [0 for _ in self.individuals]

        # ====== initialize the individual for interacting with environment ======
        self.BRI = ind_REGISTRY[args.ind](self.args, self.pp, self)
        if hasattr(self.args, 'BRI_load_path'):
            self.BRI.load_individual(self.args.BRI_load_path)

    def run(self):
        ''' Randomly choose a teammate and interact with the environment to collect trajectories.
            This version only focuses on data collection, no training is performed.
        '''
        global_start_time = time.time()
        done = False
        count = 0
        last_save = 0
        while count < self.args.t_max:
            self.logger.console_logger.info(f'================ MetaEpoch: {count} ================')
            self.logger.console_logger.info(f"Time passed: {time_str(time.time() - global_start_time)}")
            self.logger.console_logger.info(f"Status: {self.status}")

            # Randomly sample a teammate
            self.teammate_id, teammate = self.sample_individual()

            # print(teammate)

            teammate_name = teammate.split('/')[-2] if '/' in teammate else teammate
            self.logger.console_logger.info(f"Chosen Teammate: {self.teammate_id}  {teammate_name}")

            # teammate = os.path.join(teammate.split('/')[-3], os.path.join(teammate.split('/')[-2], teammate.split('/')[-1]))
            # Set the randomly selected teammate for BRI to interact with
            # self.BRI.set_agents(teammate)
            self.BRI.runner.mac.load_models(teammate)
            self.BRI.runner.setup(self.BRI.scheme, self.BRI.global_groups, self.BRI.preprocess, self.BRI.mac)

            # Instead of training, collect trajectories by interacting with the environment
            done = self.BRI.collect_trajectories()  # Changed to collect trajectories

            # Log the interaction status (e.g., average return, etc.)
            self.status[self.teammate_id] = self.BRI.status['test_return_mean']
            count += self.args.batch_size_run

            # Periodically save the collected trajectories
            if self.args.save_BR and (self.BRI.episode - last_save >= self.args.save_BR_episodes or done):
                self.BRI.save_trajectories()  # Changed to save collected trajectories
                self.BRI.init_buffer()
                last_save = self.BRI.episode

        # Close the environment after collection
        self.BRI.close_env()


