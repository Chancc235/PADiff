import torch
import torch.nn as nn

class CoGoal(nn.Module):
    def __init__(self, hidden_dim, num, state_dim=75):
        super(CoGoal, self).__init__()
        
        self.fc1 = nn.Linear(state_dim * num + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim * num)
        self.num = num
        self.state_dim = state_dim
        
        self.activation = nn.LeakyReLU(0.01)
        self.out_activation = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)


    def forward(self, state, r_hat):
        state = state.view(state.shape[0], -1)
        x = torch.cat([state, r_hat], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.out_activation(x)
        x = x.view(-1, self.num, self.state_dim) # (batch_size, num_agents, state_dim)
        
        return x