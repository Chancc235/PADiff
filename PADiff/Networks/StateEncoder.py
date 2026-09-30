import torch
import torch.nn as nn

class StateEncoder(nn.Module):

    def __init__(self, state_dim, num_agents, embed_dim, seq_len, num_heads=2, hidden_dim=256):
        super(StateEncoder, self).__init__()
        self.num_agents = num_agents
        self.state_embed = nn.Linear(state_dim*seq_len, hidden_dim)
        self.obs_embed = nn.Linear(state_dim*seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, embed_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(0.01)
        
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs, states):

        batch_size = obs.size(0)
        obs_embed = self.obs_embed(obs.view(batch_size, -1)).unsqueeze(1)

        states_embed = self.state_embed(states.view(batch_size, self.num_agents, -1))
        states_embed = torch.cat([states_embed, obs_embed], dim=1)
        # cross attention
        attn_output, _ = self.cross_attention(obs_embed, states_embed, states_embed)
        attn_output = self.attn_layer_norm(attn_output)
        
        x = self.fc2(attn_output)
        x = self.activation(x)
        x = self.layer_norm(x)
        
        x = self.fc3(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        
        output = self.output_layer(x)
        output = output.squeeze(1)
        mean = output[..., :output.size(1) // 2] 
        log_var = output[..., output.size(1) // 2:]  
 
        return mean, log_var
