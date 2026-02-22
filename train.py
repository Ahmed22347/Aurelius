import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import config
import torch.nn.functional as F
from solvers.nearest_neighbor import solve_nn
from models.attention_model import AttentionTSP
from rl.sampler import sample_tour
from data.dataset import LondonTSPDataset
from utils.helpers import collate_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train():

    model = AttentionTSP(
        embed_dim=config.EMBED_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    dataset = LondonTSPDataset(size=1000)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    critic_weight = 0.1  # keep critic influence small

    for epoch in range(config.EPOCHS):

        epoch_loss = 0
        epoch_cost = 0

        for node_features, C, padding_mask in loader:

            node_features = node_features.to(device)
            C = C.to(device)
            padding_mask = padding_mask.to(device)

            log_probs, rewards, values = sample_tour(
                model, node_features, C, padding_mask
            )

            costs = -rewards.detach()

            # ---- Normalize rewards for stability ----
            reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # ---- Advantage ----
            advantage = reward_norm - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # ---- Actor loss ----
            actor_loss = -(advantage.detach() * log_probs).mean()

            # ---- Critic loss ----
            critic_loss = F.mse_loss(values, reward_norm.detach())

            loss = actor_loss + critic_weight * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cost += costs.mean().item()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Loss: {epoch_loss/len(loader):.4f} | "
            f"Avg Cost: {epoch_cost/len(loader):.2f}"
        )

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "model_checkpoint.pt")

    torch.save(model.state_dict(), "model_final.pt")


if __name__ == "__main__":
    train()