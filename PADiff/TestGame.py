from envs.stag_hunt.stag_hunt import StagHunt
from envs.lbf.foraging import ForagingEnv
from envs.overcooked.overcookedenv import OvercookedMultiEnv
import numpy as np
import yaml
import random
import os
import torch as th
from controllers import REGISTRY as mac_REGISTRY
from types import SimpleNamespace
from components.episode_buffer import MetaReplayBuffer
from components.transforms import OneHot
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Test:
    def __init__(self, env_type:str, random=False,):

        self.env_type = env_type
        self.random = random
        if self.env_type == 'PP4a':
            self.env_name ='stag_hunt'
            self.test_yaml = 'test_PP.yaml'
            self.teammate_model_path = f'../saves/PP4a/PP4a_test_models/4/'
        elif self.env_type == 'LBF':
            self.env_name ='lbf'
            self.test_yaml = 'test_LBF.yaml'
            self.teammate_model_path = f'../saves/LBF/LBF_test_models/4/'
        elif self.env_type == 'overcooked':
            self.env_name ='overcooked'
            self.test_yaml = 'test_overcooked.yaml'
            self.teammate_model_path = f'../saves/overcooked/overcooked_test_models/4/'
            # self.teammate_model_path = f'../saves/overcooked/overcooked_test_models/8/'
        teammate_list = []
        for file_name in os.listdir(self.teammate_model_path):
            file_path = os.path.join(self.teammate_model_path, file_name)
            teammate_list.append(file_path)
        self.teammate_list = teammate_list

    
    def load_args_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            args = yaml.safe_load(f)
        return args

    def merge_dicts(self, base_dict, custom_dict):
        for key, value in custom_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self.merge_dicts(base_dict[key], value)
            else:
                base_dict[key] = value

    def init_game_setting(self):
        env_config_path = f'../src/config/envs/{self.env_name}.yaml'
        args_dict = self.load_args_from_yaml(f'../src/config/{self.test_yaml}')
        test_dict = args_dict
        default_dict = self.load_args_from_yaml('../src/config/default.yaml')
        env_args = self.load_args_from_yaml(env_config_path)

        game_args = env_args['env_args']
        game_args['seed'] = random.randint(1, 10000)
        if self.env_type == "PP4a":
            self.episode_limit = game_args['episode_limit']
        if self.env_type == "LBF":
            self.episode_limit = 50
        if self.env_type == "overcooked":
            self.episode_limit = 400

        self.merge_dicts(args_dict, default_dict)
        self.merge_dicts(args_dict, game_args)

        args = SimpleNamespace(**args_dict)

        args.agent_output_type = "q"
        args.device = "cuda"

        if self.env_name =='stag_hunt':
            self.env = StagHunt(**game_args)
            args.n_actions = self.env.n_actions
        if self.env_name == 'lbf':
            self.env = ForagingEnv(**game_args)
            args.n_actions = self.env.n_actions
        if self.env_name == 'overcooked':
            self.env = OvercookedMultiEnv(**game_args)
            args.n_actions = self.env.action_space.n
        self.n_actions = args.n_actions
        env_info = self.env.get_env_info()
        args.n_agents = self.env.n_agents
        # print(args)
        groups = {
            "agents": args.n_agents
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
            "agents": args.n_agents
        }
        self.buffer = MetaReplayBuffer(scheme, global_groups, 1024, env_info["episode_limit"] + 1,
                                    preprocess=preprocess,
                                    device=args.device)
        if self.env_type == 'LBF':
            args.max_food = test_dict['env_args']['max_food']
            args.field_size = test_dict['env_args']['field_size']
            args.sight = test_dict['env_args']['sight']
            args.population_alg = 'vdn'

        self.mac = mac_REGISTRY[args.mac](self.buffer.scheme, groups, args)
        self.new_batch = partial(EpisodeBatch, self.buffer.scheme, groups, 1, self.episode_limit + 1,
                                    preprocess=preprocess, device=args.device)
 

    def test_game_dif(self, test_episodes, agent, K=1):
        self.init_game_setting()

        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        
        env = self.env
        episode = 0
        return_list = []
        with tqdm(total=len(range(test_episodes)), desc="testing") as pbar:
            for _ in range(test_episodes):

                teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
                self.mac.load_models(self.teammate_list[teammate_model_idx])
                self.mac.init_hidden(batch_size=1)
                batch = self.new_batch()
                obs, state = env.reset()
                avail_actions = env.get_avail_actions()
                
                pre_transition_data = {
                    "state": [[state]],
                    "avail_actions": [[avail_actions]],
                    "obs": [[obs]]
                }
                batch.update(pre_transition_data, ts=0)
                
                done = False
                step_count = 0
                total_reward = 0
                teammate_idx = random.randint(0, self.env.n_agents - 1)

                o_list = [torch.zeros(obs[0].shape[-1]).unsqueeze(0) for i in range(K)]
                s_list = [torch.zeros(len(obs), obs[0].shape[-1]).unsqueeze(0) for i in range(K)]

                while not done and step_count < self.episode_limit:
                    if self.env_type == "overcooked":
                        dynamic_env_infos = env.get_dynamic_env_info()
                    
                    if self.env_type == "overcooked":
                        actions_tensor = self.mac.select_actions(batch, bs=[0], t_ep=step_count, t_env=1, test_mode=True, dynamic_env_infos=dynamic_env_infos)
                    else:
                        actions_tensor = self.mac.select_actions(batch, bs=[0], t_ep=step_count, t_env=1, test_mode=True)

                    actions = actions_tensor[0].numpy()
                    
                    o_list.append(torch.tensor(obs[teammate_idx]).unsqueeze(0))
                    s_list.append(torch.tensor(np.array(obs)).unsqueeze(0))
                    if len(o_list) > K:
                        o_list = o_list[-K:]
                        s_list = s_list[-K:]

                    if self.random:
                        action_ad = agent.take_action()
                    else:
                        action_ad = agent.take_action(o_list, s_list, self.n_actions)
                    
                    action_ad = torch.tensor(action_ad)
                    index = torch.argmax(action_ad, dim=-1)
                    actions[teammate_idx] = index
                    one_hot_action = torch.zeros_like(action_ad) 
                    one_hot_action.scatter_(-1, index, 1) 

                    actions_chosen = {
                        "actions": actions_tensor[0].unsqueeze(1).to("cuda")
                    }
                    batch.update(actions_chosen, bs=[0], ts=step_count, mark_filled=False)


                    reward, done, info = env.step(actions)
                    state = env.get_state()
                    obs = env.get_obs()
                    avail_actions = env.get_avail_actions()

                    if not done:
                        pre_transition_data = {
                            "state": [[state]],
                            "avail_actions": [[avail_actions]],
                            "obs": [[obs]]
                        }
                        batch.update(pre_transition_data, ts=step_count + 1)

                    post_transition_data = {
                        "reward": [[reward]],
                        "terminated": [[done]]
                    }
                    batch.update(post_transition_data, bs=[0], ts=step_count, mark_filled=False)

                    total_reward += reward
                    step_count += 1

                episode += 1
                return_list.append(total_reward)
                pbar.update(1)
            
        print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)
