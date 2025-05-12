import time

from meta.individual import REGISTRY as ind_REGISTRY
from meta.population import StrPopulation
from utils.config_utils import update_args
from utils.timehelper import time_str


class Stage3Population(StrPopulation):
    '''we do not keep a list of individuals, instead we keep a list of 'model_save_paths'
       Since only BRIndividual actually trains here, we leave most of the training implementations
       to BRI and simply calls its methods here.
    '''

    def __init__(self, args, global_logger) -> None:
        super().__init__(args, global_logger)
        self.args = update_args(self.args, self.args.target_alg)

        # ====== record training status ======
        self.status = [0 for _ in self.individuals]

        # ====== initialize the only individual that actually interacts with env ======
        self.BRI = ind_REGISTRY[args.ind](self.args, self.pp, self)
        if hasattr(self.args, 'BRI_load_path'):
            self.BRI.load_individual(self.args.BRI_load_path)

    def run(self):
        ''' Random choose teammate and train target agent with it.
            If target agent reaches timestep limit, return done=True to indicate end of experinment.
        '''

        global_start_time = time.time()
        done = False
        count = 0
        last_save = 0
        while not done:
            self.logger.console_logger.info(f'================ MetaEpoch: {count} ================')
            self.logger.console_logger.info(f"Time passed: {time_str(time.time() - global_start_time)}")
            self.logger.console_logger.info(f"Status:{self.status}")

            # prioritized sample teammate
            self.teammate_id, teammate = self.sample_individual()
            teammate_name = teammate.split('/')[-2] if '/' in teammate else teammate
            self.logger.console_logger.info(f"Choose:{self.teammate_id}  {teammate_name}")

            # training with sampled teammate
            self.BRI.set_agents(teammate)
            done = self.BRI.train()
            # update status
            self.status[self.teammate_id] = self.BRI.status['return_mean']
            count += 1

            if self.args.save_BR and (self.BRI.episode - last_save >= self.args.save_BR_episodes or done or last_save == 0):
                self.BRI.save_individual()
                last_save = self.BRI.episode

        self.BRI.close_env()
