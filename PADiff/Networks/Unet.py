import torch
import torch.nn as nn
import torch.nn.functional as F
from .Diffusion_models.helpers import SinusoidalPosEmb

class TimestepEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        # t: (batch_size,)  -> (batch_size, 1)
        t = t.unsqueeze(-1).float()
        return self.embed(t)  # (batch_size, embed_dim)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups=4, residual=True, downsample=False):
        super().__init__()
        self.residual = residual
        self.downsample = downsample
        
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm1 = nn.GroupNorm(norm_groups, out_channels)
        self.act1 = nn.Mish()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)
        self.act2 = nn.Mish()
        
        if residual:
            if in_channels != out_channels or downsample:
                self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.res_conv = nn.Identity()
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.residual:
            residual = self.res_conv(residual)
            out = out + residual
        out = self.act2(out)
        return out

class UNetModel(nn.Module):
    def __init__(self, embed_dim, action_dim, base_channels=16, norm_groups=2):
        super().__init__()
        self.state_proj = nn.Linear(embed_dim, base_channels)
        self.action_proj = nn.Linear(action_dim, base_channels)
        self.time_embed = TimestepEmbed(base_channels)

        self.initial_expand = nn.Conv1d(2 * base_channels, base_channels * 2, kernel_size=3, padding=1)

        # Downsampling
        self.down1 = ConvBlock1D(base_channels*2, base_channels*4, norm_groups=norm_groups, downsample=True)
        self.down2 = ConvBlock1D(base_channels*4, base_channels*8, norm_groups=norm_groups, downsample=True)

        # Bottleneck
        self.bottleneck = ConvBlock1D(base_channels*8, base_channels*8, norm_groups=norm_groups, downsample=False)

        # Upsampling
        self.upconv2 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.upblock2 = ConvBlock1D(base_channels*8 + base_channels*4, base_channels*4, norm_groups=norm_groups, downsample=False)

        self.upconv1 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.upblock1 = ConvBlock1D(base_channels*4 + base_channels*2, base_channels*2, norm_groups=norm_groups, downsample=False)

        self.final_conv = nn.Conv1d(base_channels*2, base_channels, kernel_size=3, padding=1)
        self.final_linear = nn.Linear(base_channels, action_dim)

    def forward(self, action, state, t):
        batch_size = state.shape[0]

        state_feat = self.state_proj(state)   # (B, base_channels)
        action_feat = self.action_proj(action) # (B, base_channels)
        t_feat = self.time_embed(t)            # (B, base_channels)

        x = torch.cat([state_feat, action_feat], dim=1)  # (B, 2*base_channels)
        x = x + torch.cat([t_feat, t_feat], dim=-1)        # (B, 2*base_channels)
        x = x.unsqueeze(-1)  # (B, 2*base_channels, 1)
        x = self.initial_expand(x)  # (B, base_channels*2, 1)
        
        # Down
        d1 = self.down1(x)  # (B, base_channels*4, 1/2)
        d2 = self.down2(d1) # (B, base_channels*8, 1/4)

        # Bottleneck
        x = self.bottleneck(d2)

        self.mid = x

        # Up
        x = self.upconv2(x)
        if x.shape[-1] != d1.shape[-1]:
            x = F.pad(x, (0, d1.shape[-1] - x.shape[-1]))
        x = torch.cat([x, d2], dim=1)
        x = self.upblock2(x)

        x = self.upconv1(x)
        if x.shape[-1] != d1.shape[-1]:
            x = F.pad(x, (0, d1.shape[-1] - x.shape[-1]))
        x = torch.cat([x, d1], dim=1)
        x = self.upblock1(x)

        x = self.final_conv(x)

        x = x.squeeze(-1)  # (B, base_channels)
        out_action = self.final_linear(x)  # (B, action_dim)

        return out_action
    
    def get_mid(self):
        return self.mid.mean(dim=-1)



class mlpModel(nn.Module):
    def __init__(self,
                 input_dim,
                 action_dim,
                 mid_dim,
                 t_dim=16
                 ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = input_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 512),
                                       nn.Mish(),
                                       nn.Linear(512, mid_dim),
                                       nn.Mish(),
                                       )

        self.final_layer = nn.Sequential(
            nn.Linear(mid_dim, 256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.ReLU()
        )

    def get_mid(self):
        return self.mid

    def forward(self, x, state, time):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        mid = self.mid_layer(x)
        self.mid = mid
        final = self.final_layer(mid)
        return final

