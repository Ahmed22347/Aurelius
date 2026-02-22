import torch

def sample_tour(model, node_features, C, padding_mask):

    encoded = model(node_features)

    # ---- Value prediction ----
    state_values = model.value(encoded)

    batch, n, _ = node_features.size()
    visited = torch.zeros(batch, n, dtype=torch.bool, device=node_features.device)
    current = torch.zeros(batch, dtype=torch.long, device=node_features.device)

    log_probs = []
    total_rewards = torch.zeros(batch, device=node_features.device)

    for _ in range(n - 1):

        mask = visited | padding_mask
        probs = model.decoder(encoded, current, mask)

        dist = torch.distributions.Categorical(probs)
        nxt = dist.sample()

        log_probs.append(dist.log_prob(nxt))

        edge_cost = C[torch.arange(batch), current, nxt]
        step_reward = -edge_cost
        total_rewards += step_reward

        visited = visited.clone()
        visited[torch.arange(batch), nxt] = True
        current = nxt

    log_probs = torch.stack(log_probs).sum(0)

    return log_probs, total_rewards, state_values