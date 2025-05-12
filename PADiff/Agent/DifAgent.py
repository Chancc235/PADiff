from Networks.dt_models.decision_transformer import DecisionTransformer
import torch.nn.functional as F
import torch
class BCdiff:
    def __init__(self, actor, env_type='PP4a'):
        self.actor = actor
        self.env_type = env_type

    def to_onehot(self, onehot_tensor, indices_to_onehot=[9, 10, 19, 23, 24]):
        onehot_tensor = onehot_tensor[..., :29]
        onehot_tensor = torch.clamp(onehot_tensor, min=0)
        indices_to_onehot = [v + i * 10 for i, v in enumerate(indices_to_onehot)]
        for index in indices_to_onehot:
            values = onehot_tensor[..., index].to(torch.int64)
            onehot_values = F.one_hot(values, num_classes=11) 
            onehot_tensor = torch.cat([onehot_tensor[..., :index], onehot_values, onehot_tensor[..., index + 1:]], dim=-1)

        return onehot_tensor.to(torch.float32)

    def take_action(self, obs, act_dim):
        self.actor.model.eval()
        action_pred = self.actor.sample_action(obs)
        return action_pred

class DifAgent:
    def __init__(self, actor, env_type='PP4a'):
        self.frame_work = actor
        self.env_type = env_type
        if env_type == "PP4a":
            self.num_agents = 4

    def to_onehot(self, onehot_tensor, indices_to_onehot=[9, 10, 19, 23, 24]):
        onehot_tensor = onehot_tensor[..., :29]
        onehot_tensor = torch.clamp(onehot_tensor, min=0)
        indices_to_onehot = [v + i * 10 for i, v in enumerate(indices_to_onehot)]
        for index in indices_to_onehot:
            values = onehot_tensor[..., index].to(torch.int64)
            onehot_values = F.one_hot(values, num_classes=11) 
            onehot_tensor = torch.cat([onehot_tensor[..., :index], onehot_values, onehot_tensor[..., index + 1:]], dim=-1)

        return onehot_tensor.to(torch.float32)

    def take_action(self, o_list, s_list, act_dim):
        self.frame_work.actor.model.eval()
        with torch.no_grad():
            if len(s_list) == 0:
                s_seq = torch.zeros(1, len(o_list), self.num_agents, o_list[0].size(-1)).to(device="cuda")
            else:
                s_seq = torch.cat(s_list, dim=0).to(device="cuda").unsqueeze(0)

            o_seq = torch.cat(o_list, dim=0).to(device="cuda").unsqueeze(0)
            if self.env_type == "LBF":
                o_seq = F.one_hot(o_seq[...,[-2, -3]].long(), num_classes=20).float()
                o_seq = o_seq.view(*o_seq.shape[:-2], -1)
                s_seq = F.one_hot(s_seq[...,[-2, -3]].long(), num_classes=20).float()   
                s_seq = s_seq.view(*s_seq.shape[:-2], -1)
            if self.env_type == "overcooked":
                o_seq = self.to_onehot(o_seq)
                s_seq = self.to_onehot(s_seq)
            z_mu, z_logvar = self.frame_work.StateEncoder(o_seq.unsqueeze(2), s_seq)
            z = torch.randn_like(z_mu) * torch.exp(z_logvar / 2) + z_mu
            action_pred = self.frame_work.sample_action(state=z)
            return action_pred