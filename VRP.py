import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import random
import time

def create_graph_from_matrix(distance_matrix, city_names):
    num_nodes = len(distance_matrix)
    G = nx.Graph()

    # Assign positions in a circle for visualization
    pos = nx.circular_layout(city_names)
    for i, city in enumerate(city_names):
        G.add_node(city, pos=pos[city])

    # Add edges only if distance is finite
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i][j] != np.inf:
                G.add_edge(city_names[i], city_names[j], weight=distance_matrix[i][j])

    return G

def route_distance(G, route):
    try:
        total = sum(G[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))
        total += G[route[-1]][route[0]]['weight']
        return total
    except KeyError:
        return np.inf

def solve_tsp_nearest_neighbor(G):
    route = [list(G.nodes())[0]]  # depot is first city
    unvisited = set(G.nodes()) - {route[0]}
    while unvisited:
        current = route[-1]
        neighbors = [x for x in unvisited if G.has_edge(current, x)]
        if not neighbors:
            return route, np.inf
        next_city = min(neighbors, key=lambda x: G[current][x]['weight'])
        route.append(next_city)
        unvisited.remove(next_city)
    return route, route_distance(G, route)

def solve_tsp_brute_force(G):
    nodes = list(G.nodes())[1:]
    min_route, min_distance = None, float('inf')
    for perm in itertools.permutations(nodes):
        route = [list(G.nodes())[0]] + list(perm)
        dist = route_distance(G, route)
        if dist < min_distance:
            min_distance, min_route = dist, route
    return min_route, min_distance

def solve_tsp_random(G, iterations=1000):
    nodes = list(G.nodes())
    min_route, min_distance = None, float('inf')
    for _ in range(iterations):
        route = [nodes[0]] + random.sample(nodes[1:], len(nodes) - 1)
        dist = route_distance(G, route)
        if dist < min_distance:
            min_distance, min_route = dist, route
    return min_route, min_distance

def solve_vrp(G, num_vehicles, solver):
    nodes = list(G.nodes())
    depot = nodes[0]
    nodes.remove(depot)
    clusters = [nodes[i::num_vehicles] for i in range(num_vehicles)]
    routes, total_distance = [], 0
    for cluster in clusters:
        cluster_nodes = [depot] + cluster
        subgraph = G.subgraph(cluster_nodes).copy()
        route, distance = solver(subgraph)

        if route is None or distance == np.inf:
            routes.append(cluster_nodes)
            total_distance += np.inf
            print(f"⚠️ Infeasible route for cluster {cluster_nodes}")
            continue

        routes.append(route)
        total_distance += route_distance(G, route)
    return routes, total_distance

def plot_vrp(G, routes, distance, title, time_taken):
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        edges = [(route[j], route[j + 1]) for j in range(len(route) - 1) if G.has_edge(route[j], route[j+1])]
        if G.has_edge(route[-1], route[0]):
            edges.append((route[-1], route[0]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2)
    edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title(f"{title}\nDistance: {distance:.2f}, Time: {time_taken:.6f}s", fontsize=14)
    plt.axis('off')
    plt.show()

def run_vrp_analysis(distance_matrix, city_names, num_vehicles):
    G = create_graph_from_matrix(distance_matrix, city_names)
    solvers = {
        "Nearest Neighbor": solve_tsp_nearest_neighbor,
        "Brute Force": solve_tsp_brute_force,
        "Random Search": solve_tsp_random
    }
    for name, solver in solvers.items():
        start_time = time.time()
        routes, distance = solve_vrp(G, num_vehicles, solver)
        elapsed_time = time.time() - start_time
        if distance == np.inf:
            print(f"{name} - No feasible solution found.")
        else:
            print(f"{name} - Routes: {routes}, Distance: {distance:.2f}, Time: {elapsed_time:.6f}s")
        plot_vrp(G, routes, distance, f"{name} Solution ({len(distance_matrix)} cities, {num_vehicles} vehicles)", elapsed_time)


# Example: 4 cities with names
city_names = ["São Paulo", "Rio de Janeiro", "Curitiba", "Belo Horizonte", "Salvador", "Fortaleza", "Recife", "Manaus"]


distance_matrix = [

	[0,	429, 405, 586, 1970, 2945, 2647, 3922],
	[429, 0, 838, 436, 1629, 2604, 2305, 3745],
    [405, 838, 0, 991, 2295, 3471, 3042, 3971],
	[586, 436, 991, 0, 1372, 2369, 2137, 3790],
	[1970, 1629, 2295, 1372, 0, 1207, 800, 4328],
	[2945, 2604, 3471, 2369, 1207, 0, 800, 4200],
	[2647, 2305, 3042, 2137, 800, 800, 0, 4138],
	[3922, 3745, 3971, 3790, 4328, 4200, 4138, 0]

]

run_vrp_analysis(distance_matrix, city_names, 3)


