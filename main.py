from data.london_generator import generate_london_graph
from solvers.nearest_neighbor import solve_nn
from solvers.two_opt import solve_two_opt

def main():

    nodes, bridges, C = generate_london_graph(
        n_nodes=50,
        n_bridges=3,
        congestion_strength=15,
        bridge_penalty=20,
        asymmetry_strength=0.1
    )

    nn_tour, nn_cost = solve_nn(C)
    opt_tour, opt_cost = solve_two_opt(nn_tour, C)

    print("NN cost:", nn_cost)
    print("2-opt cost:", opt_cost)


if __name__ == "__main__":
    main()