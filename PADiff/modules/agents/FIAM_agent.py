import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, hidden = self.lstm(x, hidden)
        h = F.relu(self.fc1(h))
        embedding = self.m_z(h).squeeze(0)
        return embedding, hidden


class Decoder(nn.Module):
    def __init__(self, input_dim1, hidden_dim, output_dim1, output_dim2):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim1)
        self.fc4 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        probs1 = F.softmax(self.fc4(h), dim=-1)

        return out, probs1


class FIAMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FIAMAgent, self).__init__()

        self.args = args
        self.state_shape = int(np.prod(args.state_shape))

        self.encoder = Encoder(self.state_shape, args.rnn_hidden_dim, args.z_dim)
        self.decoder = Decoder(args.z_dim, args.rnn_hidden_dim, input_shape *
                               self.args.n_ally_agents, args.n_actions*self.args.n_ally_agents)

        self.fc1 = nn.Linear(input_shape + self.args.z_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def init_hidden(self, bs):
        h_encoder = [th.zeros([1, bs, self.args.rnn_hidden_dim], device=self.args.device).contiguous(),
                     th.zeros([1, bs, self.args.rnn_hidden_dim], device=self.args.device).contiguous()]
        h_agent = th.zeros([bs*self.args.n_agents, self.args.rnn_hidden_dim], device=self.args.device).contiguous()

        return h_encoder, h_agent

    def forward(self, input_dict, hidden, **kwargs):
        obs = input_dict['obs']
        state = input_dict["state"]
        bs = state.shape[0]

        # ============ do modeling ============
        z, h_encoder = self.encoder(state, hidden[0])
        z = z.unsqueeze(1).expand(-1, self.args.n_agents, -1).reshape(bs*self.args.n_agents, -1)

        # ============ calculate extended value ============
        x = F.relu(self.fc1(th.cat([obs, z.clone().detach()], dim=-1)))
        h_agent = self.rnn(x, hidden[1])
        q = self.fc2(h_agent)
        output_dict = {"q": q}

        # ============ calculate additional loss ============
        if 'train_mode' in kwargs and kwargs['train_mode']:
            obs_loss, act_loss = self.eval_decoding(z, input_dict)
            output_dict["losses"] = {"obs_loss": obs_loss, "act_loss": act_loss}

        if 'evaluate_accuracy' in kwargs and kwargs['evaluate_accuracy']:
            # TODO: temporary use, for test figure
            output_dict["recon_accurate"] = self.calculate_accurate_reconstruction(z, input_dict)

        return output_dict, [h_encoder, h_agent]

    def eval_decoding(self, z, input_dict):
        modelled_obs, modelled_act = input_dict['teammate_obs'], input_dict['teammate_actions']
        bs = modelled_obs.shape[0]

        out, probs = self.decoder(z)
        out = out.reshape(bs, self.args.n_agents, self.args.n_ally_agents, -1)
        probs = probs.reshape(bs*self.args.n_agents*self.args.n_ally_agents, -1)
        modelled_obs = modelled_obs.reshape(bs, 1, self.args.n_ally_agents, -1).expand(-1, self.args.n_agents, -1, -1)
        modelled_act = modelled_act.reshape(bs, 1, self.args.n_ally_agents, -1).expand(-1, self.args.n_agents, -1, -1).flatten()

        obs_loss = self.mse(out, modelled_obs)
        act_loss = self.ce(probs, modelled_act)

        return obs_loss, act_loss

    def calculate_accurate_reconstruction(self, z, input_dict):
        modelled_act = input_dict['teammate_actions']
        bs = modelled_act.shape[0]
        modelled_act = modelled_act.reshape(bs, 1, self.args.n_ally_agents).expand(-1, self.args.n_agents, -1)

        _, probs = self.decoder(z)
        probs = probs.reshape(bs, self.args.n_agents, self.args.n_ally_agents, -1)

        recon_accurate = th.logical_or(probs.argmax(dim=-1) == modelled_act, modelled_act == 0).reshape(bs, -1).float().mean(-1)
        return recon_accurate   # pytorch tensor

    def save_models(self, path):
        th.save(self.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
