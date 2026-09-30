import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
import  torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class ALTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.5, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # self.mlp0 = Mlp(in_features=hidden_size, hidden_features=hidden_size, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, c):
        shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.mlp0(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + modulate(self.mlp(modulate(self.norm1(x), shift1, scale1)), shift2, scale2)
        x = self.dropout(x)
        return x

class FinalLayer(nn.Module):

    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)


    def forward(self, x, c):

        x = self.norm_final(x)
        x = self.linear(x)  # (batch_size, action_dim)
        x = F.relu(x)
        return x

class ALT(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        seq_len,
        agent_num,
        embed_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        dropout_rate=0.5
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.agent_num = agent_num
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        # State embedding
        self.state_embed = nn.Linear(embed_dim, hidden_size)
        self.x_embed = nn.Linear(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Position embedding using sin-cos encoding
        position = torch.arange(1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, hidden_size)
        pe[0, 0::2] = torch.sin(position * div_term)
        pe[0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embed', pe)

        self.blocks = nn.ModuleList([
            ALTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate) for _ in range(depth // 2)
        ])

        self.blocks2 = nn.ModuleList([
            ALTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth - depth // 2)
        ])
        self.final_layer = FinalLayer(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


    def forward(self, x, states, t, use_condition=True, test=False):
        """
        Forward pass of ALT for policy learning.
        states: (batch_size, state_dim) tensor of state inputs(z)
        t: (batch_size,) tensor of diffusion timesteps
        """
        batch_size = states.shape[0]
        # Reshape and embed states
        y = states
        if use_condition:
            y = self.state_embed(y)  # (batch_size, hidden_size)
        else:
            y = torch.zeros(batch_size, self.hidden_size, 
                          device=states.device, dtype=states.dtype)
        
        x = self.x_embed(x)  # (batch_size, hidden_size)
        t = self.t_embedder(t)  # (batch_size, hidden_size)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        if not test:
            self.mid = x
        for block in self.blocks2:
            x = block(x, c)

        x = x.mean(dim=1)
        action_noise = self.final_layer(x, t)
        
        return action_noise

    def forward_with_cfg(self, x, states, t, cfg_scale=3.0, test=False):
        noise_uncond = self.forward(x, states, t, use_condition=False, test=test)
        noise_cond = self.forward(x, states, t, use_condition=True, test=test)
        
        return noise_uncond + cfg_scale * (noise_cond - noise_uncond)

    def get_mid(self):
        return self.mid.mean(dim=1)
