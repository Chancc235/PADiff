import torch as th
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class GRUVIEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, args):
        super(GRUVIEncoder, self).__init__()

        self.args = args

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output

        self.fc1 = nn.Linear(d_input, d_hidden)
        self.rnn = nn.GRUCell(d_hidden, d_hidden)
        self.mean = nn.Linear(d_hidden, d_output)
        self.var = nn.Linear(d_hidden, d_output)

    def init_hidden(self, bs):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.d_hidden).zero_().expand(bs, -1).contiguous()

    def forward(self, inputs, hidden_state, test_mode=False):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.d_hidden)
        h = self.rnn(x, h_in)

        if test_mode:
            z = self.mean(h)
        else:
            mean = self.mean(h)
            var = th.clamp(th.exp(self.var(h)), min=0.002)
            dis = D.Normal(mean, var**(1/2))
            z = dis.rsample()
        
        if self.args.normalize_z:
            z = F.normalize(z, p=2, dim=1)

        return z, h
