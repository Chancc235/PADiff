import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, episodes_data):
        
        self.episodes_data = episodes_data
    
    def __len__(self):
        return len(self.episodes_data)
    
    def __getitem__(self, idx):
        episodes_data = self.episodes_data[idx]
        
        return episodes_data

class NewCustomDataset(Dataset):
    def __init__(self, episodes_data):
        
        data = []
        for ep in episodes_data:
            for i in range(len(ep["obs"])):
                data.append({"obs": ep["obs"][i], "action": ep["action"][i], "rtg": ep["rtg"][i]})
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        slice_data = self.data[idx]
        
        return slice_data