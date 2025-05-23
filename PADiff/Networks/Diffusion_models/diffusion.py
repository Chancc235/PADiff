import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .helpers import (cosine_beta_schedule,
                        linear_beta_schedule,
                        vp_beta_schedule,
                        extract,
                        Losses)
from .utils.utils import Progress, Silent


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 schedule_type='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 cfg_scale=3.0): 
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.model_frozen = copy.deepcopy(self.model)
        self.cfg_scale = cfg_scale  

        if schedule_type == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif schedule_type == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, grad=True):
        if grad:
            # noise = self.model.forward_with_cfg(x, s, t, False, test=True)
            noise = self.model(x=x, states=s, t=t, use_condition=True, test=True)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model_frozen(x, s, t))

        if self.clip_denoised:
            x_recon.clamp_(0, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s, grad=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, grad=grad)
        noise = torch.randn_like(x) / 3
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = F.relu(x) 
        x = F.softmax(x, dim=-1) 

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        action = F.relu(action)
        action = F.softmax(action, dim=-1)
        return action


    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start) / 3

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=None):
        noise = torch.randn_like(x_start.float()) / 3

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # uncond_noise_pred = self.model(x_noisy, state, t, False)
        cond_noise_pred = self.model(x_noisy, state, t, True)

        if self.predict_epsilon:
            cond_loss = self.loss_fn(cond_noise_pred, noise, weights)
            # uncond_loss = self.loss_fn(uncond_noise_pred, noise, weights)
        else:
            cond_loss = self.loss_fn(cond_noise_pred, x_start, weights)

        loss = cond_loss
        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

    def step_frozen(self):
        for param, target_param in zip(self.model.parameters(), self.model_frozen.parameters()):
            target_param.data.copy_(param.data)

    def sample_t_middle(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = F.relu(x) 
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

        t = np.random.randint(0, int(self.n_timesteps*0.2))
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, grad=(i == t))
        

        x = F.relu(x)
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        return x

    def sample_t_last(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        x = torch.randn(shape, device=device)
        x = F.relu(x) 
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)  

        cur_T = np.random.randint(int(self.n_timesteps * 0.8), self.n_timesteps)
        for i in reversed(range(0, cur_T)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i != 0:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)
            else:
                x = self.p_sample(x, timesteps, state)

        x = F.relu(x)
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        return x

    def sample_last_few(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        x = torch.randn(shape, device=device)
        x = F.relu(x)  
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

        nest_limit = 5
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i >= nest_limit:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)
            else:
                x = self.p_sample(x, timesteps, state)

        x = F.relu(x)
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        return x
