import numpy as np
import torch as th
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, output_dim)
        self.var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        h = self.rnn(x, hidden)
        mean = self.mean(h)
        var = self.var(h)

        return mean, var, h


class Decoder(nn.Module):
    def __init__(self, input_dim1, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)

        return out


class ODITSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ODITSAgent, self).__init__()

        self.args = args
        self.state_shape = int(np.prod(args.state_shape))

        self.team_encoder = Encoder(self.state_shape, args.rnn_hidden_dim, args.z_dim)
        self.team_decoder = Decoder(args.z_dim, args.rnn_hidden_dim, self.state_shape)

        self.proxy_encoder = Encoder(input_shape, args.rnn_hidden_dim, args.z_dim)
        self.proxy_decoder = Decoder(args.z_dim, args.rnn_hidden_dim, args.z_dim)

        self.fc1 = nn.Linear(input_shape + args.z_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self, bs):
        h_team_encoder = th.zeros([bs, self.args.rnn_hidden_dim], device=self.args.device).contiguous()
        h_proxy_encoder = th.zeros([bs*self.args.n_agents, self.args.rnn_hidden_dim], device=self.args.device).contiguous()
        h_agent = th.zeros([bs*self.args.n_agents, self.args.rnn_hidden_dim], device=self.args.device).contiguous()

        return h_team_encoder, h_proxy_encoder, h_agent

    def forward(self, input_dict, hidden, test_mode=False, **kwargs):
        obs = input_dict['obs']
        h_team_encoder = hidden[0]

        # ============ do modeling ============
        mean, var, h_proxy_encoder = self.proxy_encoder(obs, hidden[1])
        if test_mode:
            z = mean
        else:
            var_clamp = th.clamp(th.exp(var), min=self.args.var_floor)
            gaussian = D.Normal(mean, var_clamp ** (1/2))
            z = gaussian.rsample()

        # ============ calculate extended value ============
        theta_M = self.proxy_decoder(z.clone().detach())
        x = F.relu(self.fc1(th.cat([obs, theta_M], dim=-1)))
        h_agent = self.rnn(x, hidden[2])
        q = self.fc2(h_agent)
        output_dict = {"q": q}

        # ============ calculate additional loss ============
        if 'train_mode' in kwargs and kwargs['train_mode']:
            mixer_input, mi_loss, h_team_encoder = self.calculate_mi_loss(mean, var, input_dict["state"], h_team_encoder)
            output_dict["mixer_input"] = mixer_input
            output_dict["losses"] = {"mi_loss": mi_loss}

        return output_dict, [h_team_encoder, h_proxy_encoder, h_agent]

    def calculate_mi_loss(self, mean_proxy, var_proxy, state, h):
        bs = state.shape[0]
        # ===== team encoder gaussian ======
        mean_team, var_team, h_team_encoder = self.team_encoder(state, h)
        var_clamp = th.clamp(th.exp(var_team), min=self.args.var_floor)
        gaussian_team = D.Normal(mean_team, var_clamp ** (1/2))
        z_team = gaussian_team.rsample()
        mixer_input = self.team_decoder(z_team)

        # ===== get what we want =====
        mi_loss = 0
        var_clamp = th.clamp(th.exp(var_proxy), min=self.args.var_floor)
        mean_proxy = mean_proxy.reshape(bs, self.args.n_agents, -1)
        var_clamp = var_clamp.reshape(bs, self.args.n_agents, -1)
        for i in range(self.args.n_agents):
            gaussian_proxy = D.Normal(mean_proxy[:, i], var_clamp[:, i] ** (1/2))
            mi_loss += kl_divergence(gaussian_proxy, gaussian_team).sum(-1).mean()

        return mixer_input, mi_loss, h_team_encoder

    def save_models(self, path):
        th.save(self.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
