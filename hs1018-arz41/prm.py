import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from environment import scene_from_file
from nearest_neighbors import nearest_neighbors, distance
from collision_checking import check_collision

def sample_config(robot_type):
    if robot_type == "arm":
        return np.random.uniform(-np.pi, np.pi, 2)
    elif robot_type == "freeBody":
        return np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-np.pi, np.pi)])

def is_collision_free(config, environment, robot_type):
    if robot_type == "arm":
        # Implement arm collision checking
        return True
    elif robot_type == "freeBody":
        robot = {'position': config[:2], 'width': 0.5, 'height': 0.5}
        return not any(check_collision(robot, obstacle) for obstacle in environment)

def build_prm(robot_type, environment, n_samples=5000, k=6):
    graph = nx.Graph()
    samples = []
    
    for _ in range(n_samples):
        config = sample_config(robot_type)
        if is_collision_free(config, environment, robot_type):
            samples.append(config)
            graph.add_node(len(samples) - 1, config=config)
    
    for i, sample in enumerate(samples):
        neighbors = nearest_neighbors(robot_type, sample, k, samples)
        for j in neighbors:
            if not graph.has_edge(i, j):
                if is_collision_free((sample, samples[j]), environment, robot_type):
                    graph.add_edge(i, j, weight=distance(sample, samples[j], robot_type))
    
    return graph, samples

def find_path(graph, start_config, goal_config, robot_type, samples):
    start_node = min(range(len(samples)), key=lambda i: distance(samples[i], start_config, robot_type))
    goal_node = min(range(len(samples)), key=lambda i: distance(samples[i], goal_config, robot_type))
    
    try:
        path = nx.shortest_path(graph, start_node, goal_node, weight='weight')
        return [samples[node] for node in path]
    except nx.NetworkXNoPath:
        return None

def visualize_prm(graph, samples, path=None, robot_type="arm"):
    plt.figure(figsize=(10, 10))
    
    pos = {i: sample[:2] for i, sample in enumerate(samples)}
    nx.draw_networkx_edges(graph, pos, alpha=0.1)
    
    plt.scatter([s[0] for s in samples], [s[1] for s in samples], c='b', s=10)
    
    if path:
        path_x, path_y = zip(*[(p[0], p[1]) for p in path])
        plt.plot(path_x, path_y, 'r-', linewidth=2)
    
    plt.title(f"PRM for {robot_type}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def animate_solution(path, robot_type, environment):
    # Implement animation of the solution path
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRM Path Planning")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Robot type")
    parser.add_argument("--start", type=float, nargs='+', required=True, help="Start configuration")
    parser.add_argument("--goal", type=float, nargs='+', required=True, help="Goal configuration")
    parser.add_argument("--map", type=str, required=True, help="Map file")
    
    args = parser.parse_args()
    
    environment = scene_from_file(args.map)
    graph, samples = build_prm(args.robot, environment)
    path = find_path(graph, args.start, args.goal, args.robot, samples)
    
    if path:
        print("Path found!")
        visualize_prm(graph, samples, path, args.robot)
        animate_solution(path, args.robot, environment)
    else:
        print("No path found.")
        visualize_prm(graph, samples, robot_type=args.robot)