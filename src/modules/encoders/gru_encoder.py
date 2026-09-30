import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, args):
        super(GRUEncoder, self).__init__()

        self.args = args

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output

        self.fc1 = nn.Linear(d_input, d_hidden)
        self.rnn = nn.GRUCell(d_hidden, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)

    def init_hidden(self, bs):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.d_hidden).zero_().expand(bs, -1).contiguous()

    def forward(self, inputs, hidden_state, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.d_hidden)
        h = self.rnn(x, h_in)
        z = self.fc2(h)
        
        if self.args.normalize_z:
            z = F.normalize(z, p=2, dim=1)

        return z, h
