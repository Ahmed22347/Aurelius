import torch

def collate_fn(batch):

    max_n = max(x[0].shape[0] for x in batch)

    padded_nodes = []
    padded_costs = []
    padding_masks = []

    for nodes, C in batch:
        n = nodes.shape[0]

        node_pad = torch.zeros(max_n, nodes.shape[1])
        node_pad[:n] = nodes

        cost_pad = torch.full((max_n, max_n), float("inf"))
        cost_pad[:n, :n] = C

        pad_mask = torch.ones(max_n, dtype=torch.bool)
        pad_mask[:n] = False

        padded_nodes.append(node_pad)
        padded_costs.append(cost_pad)
        padding_masks.append(pad_mask)

    return (
        torch.stack(padded_nodes),
        torch.stack(padded_costs),
        torch.stack(padding_masks),
    )