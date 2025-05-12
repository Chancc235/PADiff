import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.logger import logger

from .discrete_diffusion import DiscreteDiffusion as Diffusion
# from .diffusion import Diffusion


class PA_Diffusion(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 model,
                 CoGoal,
                 CoReturn,
                 StateEncoder,
                 lr,
                 beta_min,
                 beta_max,
                 optimizer="AdamW",
                 schedule_type="absorbing",
                 n_timesteps=100,
                 lambda_aux=0.01,
                 cfg_scale=3.0
                 ):
        self.CoGoal = CoGoal
        self.CoReturn = CoReturn
        self.StateEncoder = StateEncoder

        self.actor = Diffusion(action_dim=action_dim, beta_min=beta_min, beta_max=beta_max, model=model, n_timesteps=n_timesteps, schedule_type=schedule_type, lambda_aux=lambda_aux, cfg_scale=cfg_scale).to(device)
        if optimizer == "AdamW":
            self.actor_optimizer = torch.optim.AdamW(list(self.actor.model.parameters()) 
                                                    + list(self.RtgNet.parameters()) 
                                                    + list(self.ReconGoal.parameters())
                                                    + list(self.StateEncoder.parameters()), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def sample_action(self, state):
        if isinstance(state, torch.Tensor):
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))

