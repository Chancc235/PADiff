import datetime
import os
import pprint
import sys
import threading
from copy import deepcopy
from os.path import abspath, dirname
from types import SimpleNamespace as SN
import json
import random
import string

import numpy as np
import torch as th
import yaml
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from meta.population import REGISTRY as pop_REGISTRY
from utils.config_utils import args_sanity_check, get_config, get_file_config, config_copy, recursive_dict_update
from utils.logging import Logger, get_logger

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    # np.random.seed(config["seed"])
    # th.manual_seed(config["seed"])
    # config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # Setup Experinment Unique Token
    unique_token = f'{args.exp_name}__{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    random_string = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    unique_token = f"{unique_token}_{random_string}"
    args.unique_token = unique_token

    # record config
    save_dir = os.path.join(args.local_results_path, args.unique_token)
    os.makedirs(save_dir, exist_ok=True)
    config_write_path = os.path.join(save_dir, "config.json")
    json_str = json.dumps(vars(args), indent=4)
    with open(config_write_path, "w") as json_file:
        json_file.write(json_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    pop = pop_REGISTRY[args.pop](args, logger)
    pop.run()

    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print(f"Thread {t.name} is alive! Is daemon: {t.daemon}")
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Get the experiment configs from "exp_config"
    exp_config = get_config(params, "--config", "exps")
    config_dict = recursive_dict_update(config_dict, exp_config)

    # Load environment default configs
    env_config = get_file_config(config_dict['env'], 'envs')
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, exp_config)    # override some env configs from exp config

    # get config from argv, such as "remark"
    def _get_argv_config(params):
        config = {}
        to_del = []
        for _v in params:
            item = _v.split("=")[0]
            if item[:2] == "--" and item not in ["envs", "algs"]:
                config_v = _v.split("=")[1]
                try:
                    config_v = eval(config_v)
                except:
                    pass
                config[item[2:]] = config_v
                to_del.append(_v)
        for _v in to_del:
            params.remove(_v)
        return config

    config_dict = recursive_dict_update(config_dict, _get_argv_config(params))

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver(file_obs_path))

    ex.run_commandline(params)
