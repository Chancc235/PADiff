import pickle

import numpy as np
import torch as th
import torch.nn as nn
from sklearn.cluster import KMeans

from .rnn_agent import RNNAgent
from .context_rnn_agent import ContextRNNAgent


class KmeansRnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(KmeansRnnAgent, self).__init__()
        self.args = args

        # k-means clustering, and KNN classifier
        self.kmeans = None  # k-means estimator

        # sub_modules
        self.sub_modules = nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.args.n_sub_modules)])
        # self.sub_modules = nn.ModuleList([ContextRNNAgent(input_shape, args) for _ in range(self.args.n_sub_modules)])

    def set_datatset(self, dataset):
        self.kmeans = KMeans(n_clusters=self.args.n_sub_modules).fit(dataset)

    def classifying_z(self, z):
        return self.kmeans.predict(z.cpu().numpy())

    def init_hidden(self, bs):
        hiddens = [sm.init_hidden(bs) for sm in self.sub_modules]
        return hiddens

    def forward(self, input_dict, hidden_state, **kwargs):
        # classify trajectory into specified control submodules
        z = input_dict["z"]
        chosen_policy_ids = self.classifying_z(z)

        # all submodules forward for the batch input, and choose only selected ones
        bs = input_dict["obs"].shape[0] // self.args.n_agents
        output_qs = th.zeros([bs, self.args.n_agents, self.args.n_actions], device=self.args.device)  # real output after masking
        h_return = []
        for id, sm in enumerate(self.sub_modules):
            hidden_sm = hidden_state[id]
            ret_sm, hidden_sm = sm(input_dict, hidden_sm)
            output_qs[chosen_policy_ids == id] = ret_sm['q'].view(bs, self.args.n_agents, -1)[chosen_policy_ids == id]
            h_return.append(hidden_sm)

        output_dict = {"q": output_qs.reshape(-1, self.args.n_actions)}
        return output_dict, h_return

    def save_models(self, path):
        th.save(self.state_dict(), f"{path}/agent.th")
        with open(f"{path}/kmeans.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

    def load_models(self, path):
        self.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
        with open(f"{path}/kmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)
