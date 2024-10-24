import os
import numpy as np
import matplotlib.pyplot as plt
import random
from component_5 import *
from component_6 import *
from component_7 import *
from component_8 import *
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

def extend(nearest_node, random_sample, step_size=0.4):
    
    direction = np.array(random_sample) - np.array(nearest_node)
    norm = np.linalg.norm(direction)
    if norm > step_size:
        direction = direction / norm * step_size  # Limit the step size
    new_config = nearest_node + direction
    return new_config

def build_rrt_star(robot_type, start_config, goal_config, environment, goal_radius=0.1, max_nodes=1000, step_size=0.4, radius=1.0):
    tree = nx.Graph()
    tree.add_node(0, config=start_config, cost=0)
    
    i = 1
    while tree.number_of_nodes() < max_nodes:
        random_sample = sample_config_rrt(robot_type, goal_config)

        configurations = [node['config'] for index, node in tree.nodes(data=True)]
        nearest_node, nearest_conf, _ = nearest_neighbors(robot_type, random_sample, configurations, 1, debug=False)[0]
        
        new_config = extend(nearest_conf, random_sample, step_size)
        
        if collision_free_conf(robot_type, new_config, environment, debug=False):
            if is_collision_free((nearest_conf, new_config), environment, robot_type):
                new_cost = tree.nodes[nearest_node]['cost'] + np.linalg.norm(nearest_conf - new_config)
                
                tree.add_node(i, config=new_config, cost=new_cost)
                tree.add_edge(nearest_node, i)
                
                nearby_nodes = nearest_neighbors(robot_type, new_config, configurations, k=len(configurations), debug=False)  # Get all nodes
                
                for nearby_node, nearby_conf, dist in nearby_nodes:
                    if dist > radius:
                        break
                    
                    cost_via_new = new_cost + np.linalg.norm(new_config - nearby_conf)
                    if cost_via_new < tree.nodes[nearby_node]['cost']:
                        tree.nodes[nearby_node]['cost'] = cost_via_new
                        tree.add_edge(i, nearby_node) 
                        if tree.has_edge(nearest_node, nearby_node):
                            tree.remove_edge(nearest_node, nearby_node) 
                
                if np.linalg.norm(new_config - goal_config) < goal_radius:
                    print(f"Goal reached after {i} nodes.")
                    return tree, i
                
                i += 1

    print(f"Maximum nodes ({max_nodes}) reached without finding the goal.")
    return tree, None



def visualize_rrt(tree, start_config, goal_config, environment, goal_radius=0.1, robot_type="arm"):
    """Visualize the RRT tree."""
    plt.figure(figsize=(10, 10))
    
    # Obstacles
    for obstacle in environment:
        obs_corners = get_corners(obstacle['position'], obstacle['width'], obstacle['height'], obstacle['orientation'])
        obs = plt.Polygon(obs_corners, edgecolor='black', facecolor='green')
        plt.gca().add_patch(obs)
    
    # Plot the tree
    for (node1, node2) in tree.edges:
        config1 = tree.nodes[node1]['config']
        config2 = tree.nodes[node2]['config']
        if robot_type == "arm":
            # Compute positions of joints for each configuration
            base = np.array([0, 0])
            joint1_1 = base + rotation_matrix(config1[0]) @ np.array([1.5, 0])  # First link length
            end_eff1 = joint1_1 + rotation_matrix(config1[0] + config1[1]) @ np.array([1.0, 0])  # Second link length

            joint1_2 = base + rotation_matrix(config2[0]) @ np.array([1.5, 0])
            end_eff2 = joint1_2 + rotation_matrix(config2[0] + config2[1]) @ np.array([1.0, 0])

            # Plot arm links for tree edge
            plt.plot([base[0], joint1_1[0], end_eff1[0]], [base[1], joint1_1[1], end_eff1[1]], 'b-')
            plt.plot([base[0], joint1_2[0], end_eff2[0]], [base[1], joint1_2[1], end_eff2[1]], 'b-')
        else:
            plt.plot([config1[0], config2[0]], [config1[1], config2[1]], 'b-')
    
    # Start and goal
    plt.scatter(start_config[0], start_config[1], c='g', marker='o', s=100, label="Start")
    plt.scatter(goal_config[0], goal_config[1], c='r', marker='x', s=100, label="Goal")
    plt.gca().add_patch(plt.Circle(goal_config[:2], goal_radius, color='red', fill=False, linestyle='--', label="Goal Region"))
    
    plt.title(f"RRT* for {robot_type}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def main(robot_type, start_config, goal_config, map_file, goal_radius):
    environment = scene_from_file(map_file)
    
    if not collision_free_conf(robot_type, start_config, environment, debug=False):
        raise ValueError("Invalid starting configuration for robot")
        
    # Build RRT
    tree, goal_node = build_rrt_star(robot_type, start_config, goal_config, environment, goal_radius=goal_radius)
    
    # Visualize the tree and solution path if found
    visualize_rrt(tree, start_config, goal_config, environment, goal_radius, robot_type)
    
    if goal_node is not None:
        # Backtrack the path from goal to start
        path = nx.shortest_path(tree, source=0, target=goal_node)
        print('Path found!')
        path_configurations = [tree.nodes[node]['config'] for node in path]
        print(path_configurations)
        animate_solution([tree.nodes[node]['config'] for node in path], robot_type, environment)

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
