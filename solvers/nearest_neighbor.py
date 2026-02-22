import numpy as np

def solve_nn(C):
    n = C.shape[0]
    visited = [False]*n
    tour = [0]
    visited[0] = True
    total_cost = 0
    current = 0

    for _ in range(n-1):
        costs = C[current].copy()
        for i in range(n):
            if visited[i]:
                costs[i] = np.inf

        nxt = np.argmin(costs)
        total_cost += C[current, nxt]
        tour.append(nxt)
        visited[nxt] = True
        current = nxt

    total_cost += C[current, 0]
    tour.append(0)

    return tour, total_cost