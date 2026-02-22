import torch
import torch.nn as nn
from .attention_encoder import AttentionEncoder
from .attention_decoder import AttentionDecoder


class AttentionTSP(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        self.encoder = AttentionEncoder(embed_dim, n_heads, n_layers)
        self.decoder = AttentionDecoder(embed_dim)

        # ---- Critic head ----
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, node_features):
        return self.encoder(node_features)

    def value(self, encoded):
        # Global graph embedding (mean pooling)
        graph_embed = encoded.mean(dim=1)
        return self.value_head(graph_embed).squeeze(-1)