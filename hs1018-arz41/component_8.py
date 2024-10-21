import os
import numpy as np
import matplotlib.pyplot as plt
import random
from component_5 import *
from component_6 import *  # Import the nearest_neighbors function
from component_7 import *
import networkx as nx

def sample_config(robot_type):
    if robot_type == "arm":
        return np.random.uniform(0, 2*np.pi, 2)
    elif robot_type == "freeBody":
        return np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), random.uniform(0, 2 * np.pi)])
    else:
        raise ValueError("Invalid robot type")

# def distance(config1, config2, robot_type):
#     if robot_type == "arm":
#         return np.linalg.norm(np.array(config1) - np.array(config2))
#     elif robot_type == "freeBody":
#         pos_dist = np.linalg.norm(np.array(config1[:2]) - np.array(config2[:2]))
#         angle_dist = min(abs(config1[2] - config2[2]), 2*np.pi - abs(config1[2] - config2[2]))
#         return pos_dist + angle_dist
#     else:
#         raise ValueError("Invalid robot type")

# def is_collision_free(config, environment, robot_type):
#     # Implement collision checking based on robot type and environment
#     # This is a placeholder and needs to be implemented based on your specific robots and collision checking method
#     return True  # Replace with actual collision checking

def build_prm(robot_type, environment, n_samples=5000, k=6):
    graph = nx.Graph()
    samples = []
    
    for _ in range(n_samples):
        config = sample_config(robot_type)
        if is_collision_free(config, environment, robot_type):
            samples.append(config)
            graph.add_node(len(samples) - 1, config=config)
    
    for i, sample in enumerate(samples):
        # Use the nearest_neighbors function from component_6.py
        neighbors = nearest_neighbors(robot_type, sample, samples, k)
        for index, conf, dist in neighbors:
            if not graph.has_edge(i, index):
                if is_collision_free((sample, samples[index]), environment, robot_type):
                    graph.add_edge(i, index, weight=dist)
    
    return graph, samples

def find_path(graph, start_config, goal_config, robot_type, samples):
    # start_node = min(range(len(samples)), key=lambda i: distance(samples[i], start_config, robot_type))
    s_node, s_conf, s_dist = nearest_neighbors(robot_type, start_config, samples, 1)
    g_node, g_conf, g_dist = nearest_neighbors(robot_type, goal_config, samples, 1)
    # goal_node = min(range(len(samples)), key=lambda i: distance(samples[i], goal_config, robot_type))
    
    try:
        path = nx.shortest_path(graph, s_node, g_node, weight='weight')
        return [samples[node] for node in path]
    except nx.NetworkXNoPath:
        return None

def visualize_prm(graph, samples, environment, path=None, robot_type="arm"):
    plt.figure(figsize=(10, 10))

    # Obstacles
    for obstacle in environment:
        rect = plt.Rectangle((obstacle['position'][0] - obstacle['width']/2, 
                              obstacle['position'][1] - obstacle['height']/2), 
                             obstacle['width'], obstacle['height'], 
                             edgecolor='black', facecolor='lightgrey')
        plt.gca().add_patch(rect)
    
    # Plot edges
    pos = {i: sample[:2] for i, sample in enumerate(samples)}
    nx.draw_networkx_edges(graph, pos, alpha=0.1)
    
    # Plot nodes
    plt.scatter([s[0] for s in samples], [s[1] for s in samples], c='b', s=10)
    
    # Plot path if available
    if path:
        path_x, path_y = zip(*[(p[0], p[1]) for p in path])
        plt.plot(path_x, path_y, 'r-', linewidth=2)
    
    plt.title(f"PRM for {robot_type}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def animate_solution(path, robot_type, environment):
    # Implement animation of the solution path
    # This is a placeholder
    pass

def main(robot_type, start_config, goal_config, map_file):
    environment = scene_from_file(map_file)
    
    graph, samples = build_prm(robot_type, environment)
    path = find_path(graph, start_config, goal_config, robot_type, samples)
    
    if path:
        print("Path found!")
        visualize_prm(graph, samples, environment, path, robot_type)
        animate_solution(path, robot_type, environment)
    else:
        print("No path found.")
        visualize_prm(graph, samples, environment, robot_type=robot_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PRM Path Planning")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Robot type")
    parser.add_argument("--start", type=float, nargs='+', required=True, help="Start configuration")
    parser.add_argument("--goal", type=float, nargs='+', required=True, help="Goal configuration")
    parser.add_argument("--map", type=str, required=True, help="Map file")
    
    args = parser.parse_args()
    
    main(args.robot, args.start, args.goal, args.map)