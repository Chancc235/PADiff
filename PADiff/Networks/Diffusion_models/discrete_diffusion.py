import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteDiffusion(nn.Module):
    """
    Discrete Diffusion model for action prediction.
    """
    def __init__(self, model, action_dim, beta_min=0.001, beta_max=0.2, cfg_scale=1,
                 n_timesteps=10, lambda_aux=0.01, schedule_type='uniform', device='cuda'):
        super().__init__()
        self.model = model
        self.action_dim = 1
        self.num_classes = action_dim
        self.num_timesteps = n_timesteps
        self.schedule_type = schedule_type
        self.device = device
        self.cfg_scale = cfg_scale
        self.lambda_aux = lambda_aux
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        self._setup_diffusion_parameters()
        self._precompute_matrices()
        
    def _setup_diffusion_parameters(self):
        # Setup noise schedule
        if self.schedule_type == 'uniform':
            self.betas = torch.linspace(self.beta_min, self.beta_max, self.num_timesteps, device=self.device)
        elif self.schedule_type == 'cosine':
            self.betas = self._get_cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        self.alphas = 1.0 - self.betas * self.num_classes
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Initialize cache for transition matrices
        self._cached_matrices = {}
        self._cache_clear_counter = 0
        
    def _precompute_matrices(self):
        """Precompute Q_t matrices and their cumulative products for efficiency"""
        cumulative_product = torch.ones((self.num_classes, self.num_classes), device=self.device)
        cumQ = []
        Q_matrices = []
        
        for t in range(self.num_timesteps):
            # Generate Q_t matrix
            Q_t = torch.full((self.num_classes, self.num_classes), self.betas[t], device=self.device)
            torch.diagonal(Q_t).copy_(self.alphas[t] + self.betas[t])
            
            if t == 0:
                cumulative_product = Q_t
            else:
                cumulative_product = Q_t @ cumulative_product
                
            cumQ.append(cumulative_product.unsqueeze(0).clone()) 
            Q_matrices.append(Q_t.unsqueeze(0))
            
        self.cumQ = torch.cat(cumQ)
        self.Q_matrices = torch.cat(Q_matrices)
        
    def _get_cosine_schedule(self):
        """Generate cosine noise schedule as in improved DDPM paper"""
        steps = self.num_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)
    
    def q_sample(self, x_0, t):
        """
        Sample from q(x_t | x_0) - forward diffusion process.
        
        Args:
            x_0: Starting actions (one-hot encoded)
            t: Timestep
        
        Returns:
            Tuple of (sampled x_t, probability distribution)
        """
        batch_size = x_0.shape[0]

        # For uniform or Gaussian diffusion
        cumQ_t = self.cumQ[t]
        prob_dist = torch.bmm(x_0.unsqueeze(-1).transpose(1, 2), 
                                cumQ_t.transpose(1, 2)).squeeze(1)
        prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)
        
        # Sample from probability distribution
        sampled_indices = torch.multinomial(prob_dist, num_samples=1)
        x_t = torch.zeros_like(x_0).scatter_(1, sampled_indices, 1)

        return x_t, prob_dist

    def _get_transition_matrix(self, t):
        """
        Get the transition matrix Q_t for q(x_t | x_0).
        
        Args:
            t: Timestep (can be a tensor or an integer)
            
        Returns:
            Transition matrix of shape [num_classes, num_classes] or [batch_size, num_classes, num_classes]
        """
        # Handle negative timesteps
        if isinstance(t, int) and t < 0:
            return torch.eye(self.num_classes, device=self.device)
        elif isinstance(t, torch.Tensor) and (t < 0).any():
            if t.dim() == 0:  # Scalar tensor
                if t.item() < 0:
                    return torch.eye(self.num_classes, device=self.device)
            else:
                assert (t >= 0).all(), "Negative timesteps not supported for batched computation"
        
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.item()
        
        if self.schedule_type == 'uniform':
            if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
                # Single timestep
                alpha_t = self.alphas_cumprod[t]
                Q_t = torch.ones((self.num_classes, self.num_classes), device=self.device) * (1 - alpha_t) / self.num_classes
                Q_t = Q_t + torch.diag(torch.ones(self.num_classes, device=self.device) * 
                                     (alpha_t + (1 - alpha_t) / self.num_classes))
                return Q_t
            else:
                # Batch of timesteps
                batch_size = t.shape[0]
                alphas_t = self.alphas_cumprod[t]  # [batch_size]
                
                diag_part = torch.zeros(batch_size, self.num_classes, self.num_classes, device=self.device)
                for i in range(self.num_classes):
                    diag_part[:, i, i] = alphas_t + (1 - alphas_t) / self.num_classes
                
                uniform_part = torch.ones(batch_size, self.num_classes, self.num_classes, device=self.device)
                uniform_part = uniform_part * (1 - alphas_t).unsqueeze(-1).unsqueeze(-1) / self.num_classes
                
                return uniform_part + diag_part
    
    def loss(self, actions, states):

        batch_size = actions.shape[0]
        
        if actions.dim() == 1 or actions.shape[1] == 1:
            x_0 = F.one_hot(actions.long().view(-1), num_classes=self.num_classes).float()
        else:
            x_0 = actions
        
        t = torch.randint(1, self.num_timesteps, (batch_size,), device=self.device)
        
        x_t, _ = self.q_sample(x_0, t)

        pred_logits = self.model(x_t, states, t)
        pred_probs = F.softmax(pred_logits, dim=-1)
        
        target_idx = torch.argmax(x_0, dim=1)
        
        aux_loss = 0.0

        q_posterior = self._fast_posterior(x_0, x_t, t)
        kl_loss = torch.sum(q_posterior * torch.log(torch.clamp(
            q_posterior / torch.clamp(pred_probs, min=1e-10), min=1e-10))) / batch_size
        
        t0 = torch.tensor([0] * batch_size, device=self.device)
        x_1, _ = self.q_sample(x_0, t0)
        x_0_logits_given_x1 = self.model(x_1, states, t0)
        nll_loss = F.cross_entropy(x_0_logits_given_x1, target_idx) / batch_size
        
        elbo_loss = kl_loss + 0.1 * nll_loss
        total_loss = elbo_loss + self.lambda_aux * aux_loss
        
        return total_loss
    
    def _fast_posterior(self, x_0, x_t, t):
        """
        Compute true posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Original clean action (one-hot)
            x_t: Noisy action at timestep t (one-hot)
            t: Current timestep
            
        Returns:
            Posterior distribution q(x_{t-1} | x_t, x_0)
        """
        x_0_idx = torch.argmax(x_0)
        x_t_idx = torch.argmax(x_t)
        
        posterior = torch.zeros(self.num_classes, device=self.device)
        
        mask_idx = self.num_classes - 1
        
        if x_t_idx == mask_idx: 
            posterior[x_0_idx] = 1.0
        else:  
            posterior[x_t_idx] = 1.0

        return posterior.squeeze(1)
    
    def sample(self, states, num_steps=None, old_x_t=None, return_trajectory=False):

        batch_size = states.shape[0]
        
        # Initialize x_T with noise
        if old_x_t is not None and num_steps is not None:
            x_t = old_x_t
        else:
            probs = torch.ones((batch_size, self.num_classes), device=self.device) / self.num_classes
            x_t_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            x_t = F.one_hot(x_t_idx, num_classes=self.num_classes).float()
            
            num_steps = self.num_timesteps

        trajectory = [x_t.clone()] if return_trajectory else None
        
        # Iterative denoising
        for timestep in reversed(range(num_steps)):
            t = torch.ones(batch_size, device=self.device).long() * timestep
            x_t, dist = self._p_sample_step(x_t, t, states)
            
            if return_trajectory:
                trajectory.append(x_t.clone())
                
        if num_steps != self.num_timesteps:
            return dist
  
        actions = torch.argmax(x_t, dim=1)
        
        if return_trajectory:
            return actions, trajectory
        
        if hasattr(self, 'return_onehot') and self.return_onehot:
            actions = F.one_hot(actions, num_classes=self.num_classes).float()
        
        return actions
    
    def _p_sample_step(self, x_t, t, states):
        """
        Perform a single denoising step: predict x_{t-1} given x_t.

        Args:
            x_t: Current noisy actions
            t: Current timestep
            states: Conditioning states

        Returns:
            Tuple of (sampled x_{t-1}, probability distribution)
        """
        pred_logits = self.model(x_t, states, t)

        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_probs = torch.clamp(pred_probs, min=1e-6)
        pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)

        return self._sample_categorical(pred_probs), pred_probs

    def _sample_categorical(self, probs):

        probs = torch.clamp(probs, min=1e-6)

        row_sums = probs.sum(dim=1, keepdim=True)
        valid_rows = row_sums > 1e-5
        
        normalized_probs = torch.zeros_like(probs)
        normalized_probs[valid_rows.squeeze(1)] = probs[valid_rows.squeeze(1)] / row_sums[valid_rows]
        
        invalid_rows = ~valid_rows.squeeze(1)
        if invalid_rows.any():
            normalized_probs[invalid_rows] = torch.ones_like(probs[invalid_rows]) / self.num_classes

        if torch.isnan(normalized_probs).any() or torch.isinf(normalized_probs).any():
            problematic_rows = torch.isnan(normalized_probs).any(dim=1) | torch.isinf(normalized_probs).any(dim=1)
            normalized_probs[problematic_rows] = torch.ones_like(probs[problematic_rows]) / self.num_classes
        
        try:
            indices = torch.multinomial(normalized_probs, num_samples=1).squeeze(1)
        except RuntimeError:
            indices = torch.argmax(normalized_probs, dim=1)
        
        return F.one_hot(indices, num_classes=self.num_classes).float()
