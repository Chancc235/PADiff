from envs.stag_hunt.stag_hunt import StagHunt
import numpy as np
import yaml
import random
import torch as th
from controllers import REGISTRY as mac_REGISTRY
from types import SimpleNamespace
from components.episode_buffer import MetaReplayBuffer
from components.transforms import OneHot
from functools import partial
from components.episode_buffer import EpisodeBatch


def worker(_):
    # 定义队友模型路径
    teammate_model_pth = '../saves/PP_models/0/5198198/'

    # 定义一个函数来加载yaml文件
    def load_args_from_yaml(yaml_file):
        with open(yaml_file, 'r') as f:
            args = yaml.safe_load(f)
        return args
    # 定义一个函数来合并两个字典
    def merge_dicts(base_dict, custom_dict):
        for key, value in custom_dict.items():
            if isinstance(value, dict) and key in base_dict:
                merge_dicts(base_dict[key], value)
            else:
                base_dict[key] = value

    # 加载环境参数
    env_name ='stag_hunt'
    env_config_path = f'./config/envs/{env_name}.yaml'

    args_dict = load_args_from_yaml('./config/test_PP.yaml')
    default_dict = load_args_from_yaml('./config/default.yaml')
    env_args = load_args_from_yaml(env_config_path)

    game_args = env_args['env_args']
    game_args['seed'] = random.randint(1, 10000)
    merge_dicts(args_dict, default_dict)
    merge_dicts(args_dict, game_args)

    args = SimpleNamespace(**args_dict)

    args.agent_output_type = "q"
    args.device = "cuda:2"

    # 初始化环境
    env = StagHunt(**game_args)
    env_info = env.get_env_info()
    args.n_actions = env.n_actions
    args.n_agents = env.n_agents

    groups = {
        "agents": game_args['n_agents']
    }

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
    }

    global_groups = {
        "agents": game_args['n_agents']
    }
    buffer = MetaReplayBuffer(scheme, global_groups, 1024, env_info["episode_limit"] + 1,
                                preprocess=preprocess,
                                device=args.device)


    # 设置 explore agent 控制器
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    # 加载模型
    mac.load_models(teammate_model_pth)
    mac.init_hidden(batch_size=1)


    if "filled" in buffer.scheme:
        del buffer.scheme["filled"]
    new_batch = partial(EpisodeBatch, buffer.scheme, groups, 1, game_args['episode_limit'] + 1,
                                preprocess=preprocess, device=args.device)

    batch = new_batch()
    obs, state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    # 开始游戏循环
    while not done and step_count < game_args['episode_limit']:

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
        }

        obs = env.get_obs()
        state = env.get_state()
        avail_actions = env.get_avail_actions()
        pre_transition_data["state"].append([state])
        pre_transition_data["avail_actions"].append([avail_actions])
        pre_transition_data["obs"].append([obs])
        batch.update(pre_transition_data, bs=[0], ts=step_count)

        # 选择动作
        if step_count == 0:
            actions = np.random.randint(0, env.n_actions, size=(env.n_agents,))
        else:
            actions_tensor = mac.select_actions(batch, bs=[0],t_ep=step_count, t_env=step_count, test_mode=True)
            
            actions = actions_tensor[0].numpy()
            '''
            actions = np.concatenate((actions[:1], np.random.randint(0, env.n_actions, size=(env.n_agents - 1,))))
            '''
        # 执行动作并获取下一步
        reward, done, info = env.step(actions)

        post_transition_data = {
            "actions": [],
            "reward": [],
            "terminated": [],
        }
        post_transition_data["actions"].append([actions])
        post_transition_data["reward"].append([reward])
        post_transition_data["terminated"].append([done])
        
        batch.update(post_transition_data, bs=[0], ts=step_count)

        # 累积奖励
        total_reward += reward
        step_count += 1

    return total_reward

from concurrent.futures import ProcessPoolExecutor


if __name__ == "__main__":
    # 使用ProcessPoolExecutor来创建并行进程
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(worker, range(16)))

    # 计算均值
    mean_result = sum(results) / len(results)
    
    print("32个程序计算的结果:", results)
    print("均值:", mean_result)

