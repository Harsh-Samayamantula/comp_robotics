import os
import numpy as np
import matplotlib.pyplot as plt
import random
from component_5 import *
from component_6 import *
from component_7 import *
import networkx as nx

def sample_config_rrt(robot_type, goal_config=None, goal_bias=0.05):
    if random.random() < goal_bias and goal_config is not None:
        return goal_config
    if robot_type == "arm":
        return np.random.uniform(0, 2*np.pi, 2)  # Two angles for arm
    elif robot_type == "freeBody":
        return np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), random.uniform(0, 2 * np.pi)])
    else:
        raise ValueError("Invalid robot type")

def extend(tree, nearest_node, random_sample, step_size=0.1):
    """Extend the tree by moving from nearest_node towards random_sample with a certain step size."""
    direction = random_sample - nearest_node
    norm = np.linalg.norm(direction)
    if norm > step_size:
        direction = direction / norm * step_size  # Limit the step size
    new_config = nearest_node + direction
    return new_config

def build_rrt(robot_type, start_config, goal_config, environment, goal_radius=0.1, max_nodes=1000):
    tree = nx.Graph()
    tree.add_node(0, config=start_config)
    
    for i in range(1, max_nodes):
        # Sample a new configuration
        random_sample = sample_config_rrt(robot_type, goal_config)
        
        # Find the nearest node in the tree
        nearest_node, nearest_conf, _ = nearest_neighbors(robot_type, random_sample, [n['config'] for _, n in tree.nodes(data=True)], 1)[0]
        nearest_config = tree.nodes[nearest_node]['config']
        
        # Extend the tree towards the sampled configuration
        new_config = extend(tree.nodes[nearest_node]['config'], random_sample)
        
        # Check if the new node is collision-free
        if is_collision_free(new_config, environment, robot_type):
            tree.add_node(i, config=new_config)
            tree.add_edge(nearest_node, i)
            
            # Check if we have reached the goal region
            if np.linalg.norm(new_config - goal_config) < goal_radius:
                print(f"Goal reached after {i} nodes.")
                return tree, i
    
    print(f"Maximum nodes ({max_nodes}) reached without finding the goal.")
    return tree, None

def visualize_rrt(tree, start_config, goal_config, environment, goal_radius=0.1, robot_type="arm"):
    """Visualize the RRT tree."""
    plt.figure(figsize=(10, 10))
    
    # Obstacles
    for obstacle in environment:
        rect = plt.Rectangle((obstacle['position'][0] - obstacle['width']/2, 
                              obstacle['position'][1] - obstacle['height']/2), 
                             obstacle['width'], obstacle['height'], 
                             edgecolor='black', facecolor='lightgrey')
        plt.gca().add_patch(rect)
    
    # Plot the tree
    for (node1, node2) in tree.edges:
        config1 = tree.nodes[node1]['config']
        config2 = tree.nodes[node2]['config']
        plt.plot([config1[0], config2[0]], [config1[1], config2[1]], 'b-')
    
    # Start and goal
    plt.scatter(start_config[0], start_config[1], c='g', marker='o', s=100, label="Start")
    plt.scatter(goal_config[0], goal_config[1], c='r', marker='x', s=100, label="Goal")
    plt.gca().add_patch(plt.Circle(goal_config[:2], goal_radius, color='red', fill=False, linestyle='--', label="Goal Region"))
    
    plt.title(f"RRT for {robot_type}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def main(robot_type, start_config, goal_config, map_file, goal_radius):
    environment = scene_from_file(map_file)
    
    # Build RRT
    tree, goal_node = build_rrt(robot_type, np.array(start_config), np.array(goal_config), environment, goal_radius=goal_radius)
    
    # Visualize the tree and solution path if found
    visualize_rrt(tree, start_config, goal_config, environment, goal_radius, robot_type)
    
    if goal_node is not None:
        # Backtrack the path from goal to start
        path = nx.shortest_path(tree, source=0, target=goal_node)
        animate_solution([tree.nodes[node]['config'] for node in path], robot_type, environment)

def animate_solution(path, robot_type, environment):
    # Placeholder for animation implementation
    pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RRT Path Planning")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Robot type")
    parser.add_argument("--start", type=float, nargs='+', required=True, help="Start configuration")
    parser.add_argument("--goal", type=float, nargs='+', required=True, help="Goal configuration")
    parser.add_argument("--goal_rad", type=float, required=True, help="Goal radius")
    parser.add_argument("--map", type=str, required=True, help="Map file")
    
    args = parser.parse_args()
    
    main(args.robot, args.start, args.goal, args.map, args.goal_rad)
