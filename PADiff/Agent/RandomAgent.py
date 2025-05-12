import numpy as np

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def take_action(self):
        return np.random.randint(0, self.n_actions, size=(1,))
