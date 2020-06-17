import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

ROOT_DIR = "/ssd/datasets/argoverse/argoverse-forecasting-dataset/"
TRAIN = ROOT_DIR + "train/data"
VAL = ROOT_DIR + "val/data"
TEST = ROOT_DIR + "test_obs/data/"
DATASET_PATH = {
        "train" : TRAIN,
        "val" : VAL,
        "test" : TEST
        }

class TrajectoryDataset(Dataset):
    
    def __init__(self, root_dir, mode):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.sequences = [(self.root_dir / x).absolute() for x in os.listdir(self.root_dir)]
        self.obs_len = 20
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = pd.read_csv(self.sequences[idx])
        agent_x = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["X"]
        agent_y = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        # return input and target
        return [agent_traj[:self.obs_len], agent_traj[self.obs_len:]]

def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
