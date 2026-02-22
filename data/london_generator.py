import numpy as np
import random

def generate_london_graph(
    n_nodes,
    n_bridges,
    congestion_strength,
    bridge_penalty,
    asymmetry_strength
):
    nodes = []

    for _ in range(n_nodes):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)

        side = 1 if y > 50 else 0
        congestion = 1 if (x-50)**2 + (y-50)**2 < 400 else 0

        nodes.append((x/100, y/100, side, congestion))

    bridges = sorted(random.uniform(10, 90) for _ in range(n_bridges))

    C = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue

            xi, yi, side_i, _ = nodes[i]
            xj, yj, side_j, cong_j = nodes[j]

            xi *= 100
            yi *= 100
            xj *= 100
            yj *= 100

            if side_i == side_j:
                base = abs(xi-xj) + abs(yi-yj)
            else:
                bridge_costs = []
                for b in bridges:
                    bridge_costs.append(
                        abs(xi-b) + abs(yi-50)
                        + bridge_penalty
                        + abs(xj-b) + abs(yj-50)
                    )
                base = min(bridge_costs)

            if cong_j:
                base += congestion_strength

            asym = random.uniform(0, asymmetry_strength)
            C[i, j] = base * (1 + asym)

    return np.array(nodes), bridges, C