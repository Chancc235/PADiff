import torch as th
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, d_input1, d_input2, d_hidden, d_output, args):
        super(MLPDecoder, self).__init__()

        self.args = args

        activation = nn.LeakyReLU()
        self.mlp = nn.Sequential(
            nn.Linear(d_input1 + d_input2, d_hidden),
            activation,
            nn.Linear(d_hidden, d_hidden),
            activation,
            nn.Linear(d_hidden, d_output)
        )

    def forward(self, bl, z, inputs):
        z = z.unsqueeze(1).expand(-1, bl, -1)
        return self.mlp(th.cat([z, inputs], dim=-1))
