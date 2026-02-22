import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(4, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, x):
        # x: (batch, n_nodes, 4)
        x = self.input_proj(x)
        x = self.encoder(x)
        return x  # (batch, n_nodes, embed_dim)