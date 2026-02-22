def count_river_crossings(tour, nodes):
    crossings = 0
    for i in range(len(tour)-1):
        if nodes[tour[i]][2] != nodes[tour[i+1]][2]:
            crossings += 1
    return crossings