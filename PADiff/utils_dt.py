import logging
import os
from datetime import datetime
import yaml
import torch
import numpy as np
import time
import random

def create_save_directory(base_dir="training_output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def setup_logger(save_dir):
    log_path = os.path.join(save_dir, "training_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

def save_config(config, save_dir):
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def print_elapsed_time(start_time, logger):
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Time passed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


def load_data(data_path, num_agents = 4):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    start_time = time.time()
    logger.info("Start load data")
    episodes_data = torch.load(data_path)
    logger.info("loaded data")
    obs, actions, states, next_states, rewards = [[] for _ in range(num_agents)], [[] for _ in range(num_agents)], [[] for _ in range(num_agents)], [[] for _ in range(num_agents)], [[] for _ in range(num_agents)]
    length = len(episodes_data)
    cnt = 0
    for episode in episodes_data:
        cnt += 1
        if cnt % 100 == 0:
            logger.info(f"load data {cnt} / {length}")
        if cnt % 1000 == 0:
            print_elapsed_time(start_time, logger)
        episode_length = np.where(episode["done"] == 1)[0]
        episode_length = episode_length[0] if len(episode_length) > 0 else episode["done"].shape[0]
        
        for i in range(num_agents):
            states[i].extend(episode["state"][:episode_length])
            next_states[i].extend(episode["next_state"][:episode_length])
            obs[i].extend(episode["state"][:episode_length, i])
            actions[i].extend(episode["action"][:episode_length, i])
            rewards[i].extend(episode["reward"][:episode_length])

    states = [np.array(s) for agent_states in states for s in agent_states]
    next_states = [np.array(n) for agent_next_states in next_states for n in agent_next_states]
    obs = [np.array(o) for agent_obs in obs for o in agent_obs ]
    actions = [np.array(a) for agent_actions in actions for a in agent_actions]
    rewards = [np.array(r) for agent_rewards in rewards for r in agent_rewards]
    logger.info(f"load data done")

    return states, next_states, obs, actions, rewards

def get_batch(episode_data, device="cuda", max_ep_len=201, max_len=20, goal_steps=1):
    obs = episode_data["obs"]
    actions = episode_data["action"]
    rewards = episode_data["reward"]
    goals = episode_data["next_state"]
    dones = episode_data["done"]

    state_dim = obs.shape[-1]
    act_dim = 1
    goal_dim = goals[0, 0, ...].shape
    batch_size = len(obs)
    s, a, g, d, timesteps, mask = [], [], [], [], [], []
    for i in range(batch_size):
        si = random.randint(0, obs.size(1) - 1 - goal_steps + 1 - max_len)
        # get sequences from dataset
        s.append(obs[i, si:si + max_len].reshape(1, -1, state_dim))
        a.append(actions[i, si:si + max_len].reshape(1, -1, act_dim))
        g.append(goals[i, si + goal_steps - 1:si + goal_steps - 1 + max_len, ...].reshape(1, -1, goal_dim[0], goal_dim[1]))
        d.append(dones[i, si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

        # padding and state
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        # s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0, a[-1]], axis=1)
        g[-1] = np.concatenate([np.zeros((1, max_len - tlen, goal_dim[0], goal_dim[1])), g[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)

        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    g = torch.from_numpy(np.concatenate(g, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)

    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, g, d, timesteps, mask


def get_batch_ok(episode_data, device="cuda", max_ep_len=201, max_len=20, goal_steps=1):
    states = episode_data["state"]
    obs = episode_data["obs"]
    actions = episode_data["action"]
    rewards = episode_data["reward"]
    goals = episode_data["next_state"]
    dones = episode_data["done"]

    state_dim = states.shape[-1]
    act_dim = 1
    goal_dim = goals[0, 0, ...].shape
    batch_size = len(states)
    s, o, a, g, d, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        si = random.randint(0, states.size(1) - 1 - goal_steps + 1 - max_len)
        # get sequences from dataset
        s.append(states[i, si:si + max_len, ...].reshape(1, -1, states[0, 0, ...].shape[0], states[0, 0, ...].shape[1]))
        o.append(obs[i, si:si + max_len].reshape(1, -1, state_dim))
        a.append(actions[i, si:si + max_len].reshape(1, -1, act_dim))
        g.append(goals[i, si + goal_steps - 1:si + goal_steps - 1 + max_len, ...].reshape(1, -1, goal_dim[0], goal_dim[1]))
        d.append(dones[i, si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

        # padding and state
        tlen = o[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, states[0, 0, ...].shape[0], states[0, 0, ...].shape[1])), s[-1]], axis=1)
        o[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), o[-1]], axis=1)
        # s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0, a[-1]], axis=1)
        g[-1] = np.concatenate([np.zeros((1, max_len - tlen, goal_dim[0], goal_dim[1])), g[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)

        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    o = torch.from_numpy(np.concatenate(o, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    g = torch.from_numpy(np.concatenate(g, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)

    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, o, a, g, d, timesteps, mask

def get_batch_dt(episode_data, device="cuda", max_ep_len=201, max_len=20, goal_steps=1):
    obs = episode_data["obs"]
    actions = episode_data["action"]
    rewards = episode_data["reward"]
    R = episode_data["rtg"]
    dones = episode_data["done"]

    state_dim = obs.shape[-1]
    act_dim = 1
    batch_size = len(obs)
    s, a, r, rtg, d, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        si = random.randint(0, obs.size(1) - 1)
        # get sequences from dataset
        s.append(obs[i, si:si + max_len].reshape(1, -1, state_dim))
        a.append(actions[i, si:si + max_len].reshape(1, -1, act_dim))
        rtg.append(R[i, si:si + max_len, ...].reshape(1, -1, 1))
        d.append(dones[i, si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

        # padding and state
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        # s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0, a[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)

        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)

    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, rtg, d, timesteps, mask


def preprocess_data(data):
    i = 0
    for item in data:
        print(i)
        i += 1
        rewards = item["reward"]
        reward_to_go = torch.tensor([sum(rewards[i:]).item() for i in range(len(rewards))])
        item["rtg"] = reward_to_go
    return data