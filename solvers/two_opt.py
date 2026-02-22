def compute_cost(tour, C):
    return sum(C[tour[i], tour[i+1]] for i in range(len(tour)-1))

def solve_two_opt(initial_tour, C):
    best = initial_tour
    best_cost = compute_cost(best, C)
    improved = True
    n = len(best)-1

    while improved:
        improved = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                if j-i == 1:
                    continue
                new = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = compute_cost(new, C)
                if new_cost < best_cost:
                    best = new
                    best_cost = new_cost
                    improved = True
        initial_tour = best

    return best, best_cost