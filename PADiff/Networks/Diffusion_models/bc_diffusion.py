# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from .diffusion import Diffusion

class Diffusion_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 model,
                 lr,
                 optimizer="Adam",
                 beta_schedule='linear',
                 n_timesteps=100):

        self.model = model 
        self.actor = Diffusion(
            state_dim=state_dim, 
            action_dim=action_dim, 
            model=self.model, 
            max_action=max_action,
            beta_schedule=beta_schedule, 
            n_timesteps=n_timesteps
        ).to(device)
        
        if optimizer == "Adam":
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        
        for _ in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            loss = self.actor.loss(action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)

        return metric

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
        path = f'{dir}/actor_{id}.pth' if id is not None else f'{dir}/actor.pth'
        torch.save(self.actor.state_dict(), path)

    def load_model(self, dir, id=None):
        path = f'{dir}/actor_{id}.pth' if id is not None else f'{dir}/actor.pth'
        self.actor.load_state_dict(torch.load(path))
