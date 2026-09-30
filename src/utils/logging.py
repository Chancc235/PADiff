from collections import defaultdict
import logging
import numpy as np
import torch as th

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    # def setup_tb(self, directory_name):
    #     # Import here so it doesn't have to be installed if you don't use it
    #     from tensorboard_logger import configure, log_value
    #     configure(directory_name)
    #     self.tb_logger = log_value
    #     self.use_tb = True

    # ! Implement multiple tbloggers by directly setting up "tbLogger" instead of using wrapped "log_value"
    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import Logger as tbLogger
        lg = tbLogger(directory_name, flush_secs=2)
        self.tb_logger = lg.log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info[f"{key}_T"].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info[f"{key}_T"] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1

            # fix bugs of original "pymarl", this part could cause problem
            tmp = []
            for x in self.stats[k][-window:]:
                if type(x[1]) == float:
                    tmp.append(x[1])
                elif type(x[1]) == th.Tensor:
                    tmp.append(x[1].cpu().numpy())

            item = "{:.4f}".format(np.mean(tmp))
            log_str += "{:<25}{:>8}".format(f"{k}:", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

