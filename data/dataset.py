import torch
from torch.utils.data import Dataset
from data.london_generator import generate_london_graph
import config
import numpy as np
import random

class LondonTSPDataset(Dataset):

    def __init__(self, size):
        self.size = size  # number of samples per epoch

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        n = random.randint(*config.TRAIN_NODE_RANGE)
        b = random.randint(*config.N_BRIDGES_TRAIN)

        nodes, bridges, C = generate_london_graph(
            n_nodes=n,
            n_bridges=b,
            congestion_strength=config.CONGESTION_STRENGTH,
            bridge_penalty=config.BRIDGE_PENALTY,
            asymmetry_strength=config.ASYMMETRY_STRENGTH
        )

        node_features = torch.tensor(nodes, dtype=torch.float32)
        cost_matrix = torch.tensor(C, dtype=torch.float32)

        return node_features, cost_matrix