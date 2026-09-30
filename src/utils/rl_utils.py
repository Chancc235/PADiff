import math
import random

import torch as th
import torch.nn as nn


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
            * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def train_test_split(item_list, test_percent, deterministic=True):

    # ----- deterministic split -----
    if deterministic:
        k = math.ceil(1 / test_percent)
        train_list, test_list = [], []
        for i, item in enumerate(item_list):
            if i % k == 0:
                test_list.append(item)
            else:
                train_list.append(item)
        return train_list, test_list
    # ----- random split -----
    else:
        n_total = len(item_list)
        offset = math.floor(test_percent * n_total)
        random.shuffle(item_list)
        return item_list[offset:], item_list[:offset]
