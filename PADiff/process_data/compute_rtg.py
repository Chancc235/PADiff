import torch
from utils_dt import preprocess_data
data_path = "data/overcooked_episodes_datas.pt"
sav_data_path = "data/overcooked_episodes_datas_rtg.pt"
data = torch.load(data_path)

torch.save(data, sav_data_path)