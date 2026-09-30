import torch.nn as nn
import torch.nn.functional as F
import torch as th


class ContextRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ContextRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape + args.z_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self, bs):
        # make hidden states on same device as model
        return th.zeros([bs, self.args.n_agents, self.args.rnn_hidden_dim], device=self.args.device).contiguous()

    def forward(self, input_dict, hidden_state, **kwargs):
        # Prepare for inputs
        obs = input_dict['obs']
        z = input_dict["z"].unsqueeze(1).expand(-1, self.args.n_agents, -1).reshape(-1, self.args.z_dim)
        inputs = th.cat([obs, z], dim=-1)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        output_dict = {"q": q}
        return output_dict, h

    def save_models(self, path):
        th.save(self.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
