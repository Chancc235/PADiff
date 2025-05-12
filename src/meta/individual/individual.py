import copy
import os
import time

from controllers import REGISTRY as mac_REGISTRY
from utils.config_utils import update_args
from utils.timehelper import time_left, time_str


class Individual:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.name = ''  # keep empty for stage2 and stage3

    def save_individual(self):
        ''' save model & status
            status describes information of current model, should include ['cur_win_rate']
        '''
        # save model
        model_save_path = os.path.join(self.args.local_results_path, self.args.unique_token,
                                       'models', self.name, str(self.runner.t_env))
        os.makedirs(model_save_path, exist_ok=True)
        self.logger.console_logger.info(f"Saving models to {model_save_path}")
        self.learner.save_models(model_save_path)

        # save status
        status_file_path = os.path.join(model_save_path, 'status.txt')
        with open(status_file_path, 'w') as f:
            for k, v in self.status.items():
                f.write(f'{k} : {str(v)}' + '\n')

    def load_individual(self, unique_token, timesteps=None, evaluate=False):
        if timesteps is None:
            path = os.path.join(self.args.local_saves_path, unique_token)
            timesteps = str(max(map(int, os.listdir(path))))
        path = os.path.join(self.args.local_saves_path, unique_token, str(timesteps))
        self.logger.console_logger.info(f"Loading models from: {path}")

        if evaluate:
            # load agent only
            self.mac.load_models(path)
        else:
            # load everything
            self.learner.load_models(path)

    def close_env(self):
        ''' close the environment '''
        self.runner.close_env()

    # ========== shared by stage2 and stage3 ==========
    def _initialize_training_time(self):
        self.episode = 0
        self.last_test_T = -self.args.test_interval - 1
        self.last_log_T = 0
        self.model_save_time = 0

        self.start_time = time.time()
        self.last_time = time.time()

        self.first_train = False

    def _test_and_log(self):
        self.logger.console_logger.info(f"t_env: {self.runner.t_env} / {self.args.t_max}")
        self.logger.console_logger.info(
            f"Estimated time left: {time_left(self.last_time, self.last_test_T, self.runner.t_env, self.args.t_max)}. Time passed: {time_str(time.time() - self.start_time)}")
        self.last_time = time.time()
        self.last_test_T = self.runner.t_env
        self.test()

    def set_agents(self, teammate):

        if len(self.alg2agent["teammate"]) <= 0:
            return
        if self.first_set:
            self._build_teammate_agents()
        # load saved model
        if self.alg2mac["teammate"].type == "network":
            self.alg2mac["teammate"].load_models(teammate)
            # set correct device
            if self.args.use_cuda:
                self.alg2mac["teammate"].cuda()
        elif self.alg2mac["teammate"].type == "rule":
            self.alg2mac["teammate"].switch_policy_type(teammate)

    def _build_teammate_agents(self):
        # update args for specific alg
        agent_args = update_args(self.args, self.args.teammate_alg)
        agent_args.agent_ids = self.alg2agent["teammate"]
        agent_args.n_agents = len(agent_args.agent_ids)

        # define mac for alg
        groups = {
            "agents": agent_args.n_agents
        }

        self.alg2mac["teammate"] = mac_REGISTRY['partial_rule'](
            self.buffer.scheme, groups, agent_args) if self.args.teammate_alg == "rule" else mac_REGISTRY['partial'](self.buffer.scheme, groups, agent_args)

        self.first_set = False

    def init_hidden(self, batch_size):
        for alg in self.alg_set:
            if len(self.alg2agent[alg]) > 0 and self.alg2mac[alg].type == "network":
                # let all macs init_hidden
                self.alg2mac[alg].init_hidden(batch_size=batch_size)

    @property
    def action_selector(self):
        return self.mac.action_selector
