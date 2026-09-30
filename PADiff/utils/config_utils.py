try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import os
from copy import deepcopy
from types import SimpleNamespace as SN

import torch as th
import yaml


def recursive_dict_update(d, u):
    if u is None:
        return d
    for k, v in u.items():
        d[k] = recursive_dict_update(d.get(k, {}), v) if isinstance(v, Mapping) else v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def reverse_dict(d, assert_unique=True):
    '''reverse key-value map, combine same key's values to a list'''
    ret = {}
    for k, v in d.items():
        if type(v) == list:
            for vv in v:
                ret[vv] = k if vv not in ret.keys() else [ret[vv], k]
        else:
            ret[v] = k if v not in ret.keys() else [ret[v], k]
    if assert_unique:
        for k, v in ret.items():
            if type(v) == list:
                assert 0, f"key {k} has multiple values: {v}"
    else:
        for k, v in ret.items():
            if type(v) != list:
                ret[k] = [ret]


def get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        return get_file_config(config_name, subfolder)


def get_file_config(file, subfolder='algs'):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", subfolder, f"{file}.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, f"{file}.yaml error: {exc}"
    return config_dict


def update_args(args, file, subfloder='algs'):
    config = vars(deepcopy(args))
    config_dict = get_file_config(file, subfloder)
    config = recursive_dict_update(config, config_dict)
    new_args = SN(**config)

    return new_args


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    if config["episodes_per_teammate"] % config["batch_size_run"] != 0:
        config["episodes_per_teammate"] = (
            config["episodes_per_teammate"]//config["batch_size_run"]) * config["batch_size_run"]
        _log.warning(f'episodes_per_teammate was changed to {config["episodes_per_teammate"]}')


    return config
