import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(LSTMEncoder, self).__init__()

        self.args = args
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.embed = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, bs):
        h_encoder = (th.zeros([1, self.hidden_dim], device=self.args.device).expand(bs, -1).reshape(1, -1, self.hidden_dim).contiguous(),
                     th.zeros([1, self.hidden_dim], device=self.args.device).expand(bs, -1).reshape(1, -1, self.hidden_dim).contiguous())

        return h_encoder

    def forward(self, x, hidden, **kwargs):
        h = F.relu(self.fc1(x))

        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        h, hidden = self.lstm(h, hidden)

        if len(h.size()) == 3:
            h = h.squeeze(0)
        h = F.relu(self.fc2(h))
        z = self.embed(h)

        if self.args.normalize_z:
            z = F.normalize(z, p=2, dim=1)

        return z, hidden
