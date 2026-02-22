import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, encoded_nodes, current_node, mask):
        # encoded_nodes: (batch, n, d)
        # current_node: (batch,)
        # mask: (batch, n)

        batch_size, n, d = encoded_nodes.size()

        # Get current node embedding
        current_embed = encoded_nodes[
            torch.arange(batch_size), current_node
        ]  # (batch, d)

        query = self.query_proj(current_embed).unsqueeze(1)  # (batch,1,d)
        keys = self.key_proj(encoded_nodes)  # (batch,n,d)

        scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(d)
        scores = scores.squeeze(1)  # (batch,n)

        scores[mask] = -1e9  # mask visited nodes

        probs = F.softmax(scores, dim=-1)

        return probs