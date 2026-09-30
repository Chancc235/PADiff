# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Diffusion_models.helpers import SinusoidalPosEmb

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 input_dim,
                 action_dim,
                 device,
                 mid_dim,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = input_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 512),
                                       nn.Mish(),
                                       nn.Linear(512, 512),
                                       nn.Mish(),
                                       nn.Linear(512, mid_dim),
                                       nn.Mish(),
                                       )

        self.final_layer = nn.Sequential(
            nn.Linear(mid_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.ReLU()
        )

    def get_mid(self):
        return self.mid

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        mid = self.mid_layer(x)
        self.mid = mid
        final = self.final_layer(mid)
        return final


