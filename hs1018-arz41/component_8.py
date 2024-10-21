import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from component_6 import load_configs, nearest_neighbors, euclidean_distance, toroidal_distance, visualize, plot_arm
from component_7 import check_collision

class Node:
    def __init__(self, config, index):
        self.config = config  # Configuration of the node (x, y, theta)
        self.index = index  # Unique identifier for the node
        self.edges = []  # Neighbors connected to this node

def generate_prm_configs(robot_type, num_configs, filename):
    """
    Wrapper to generate configurations based on the robot type
    """
    if robot_type == 'freeBody':
        from component_7 import generate_freebody_configs
        generate_freebody_configs(num_configs, filename)
    else:  # robot_type == 'arm'
        from component_7 import generate_arm_configs
        generate_arm_configs(num_configs, filename)

def is_collision_free(config1, config2, environment, robot_type):
    """
    Check if the straight path between config1 and config2 is collision-free.
    """
    num_steps = 10  # Number of interpolated steps between the two configurations
    for i in range(num_steps + 1):
        # Interpolate between the two configurations
        t = i / num_steps
        if robot_type == 'freeBody':
            intermediate_config = (
                config1[0] * (1 - t) + config2[0] * t,
                config1[1] * (1 - t) + config2[1] * t,
                config1[2] * (1 - t) + config2[2] * t
            )
            robot = {'position': (intermediate_config[0], intermediate_config[1]), 'width': 0.5, 'height': 0.3}
        else:  # robot_type == 'arm'
            intermediate_config = (
                config1[0] * (1 - t) + config2[0] * t,
                config1[1] * (1 - t) + config2[1] * t
            )
            robot = {'position': (0, 0), 'theta': intermediate_config}  # Arm is different, we can assume it's at origin
            
        # Use component_6's check_collision function
        if any(check_collision(robot, obs) for obs in environment):
            return False
    return True

def build_prm(configs, neighbors, environment, robot_type):
    """ Build the PRM with connections between nearest neighbors if collision-free """
    nodes = [Node(config, i) for i, config in enumerate(configs)]
    
    for i, config in enumerate(configs):
        for j in neighbors[i]:
            if is_collision_free(config, configs[j], environment, robot_type):
                nodes[i].edges.append(nodes[j])
                nodes[j].edges.append(nodes[i])  # Bidirectional edge
    
    return nodes

def dijkstra_search(start_node, goal_node):
    """ Perform Dijkstra's algorithm to find the shortest path from start to goal """
    import heapq
    queue = [(0, start_node)]
    distances = {start_node.index: 0}
    predecessors = {start_node.index: None}
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_node == goal_node:
            break
        
        for neighbor in current_node.edges:
            distance = current_distance + np.linalg.norm(np.array(current_node.config[:2]) - np.array(neighbor.config[:2]))
            if neighbor.index not in distances or distance < distances[neighbor.index]:
                distances[neighbor.index] = distance
                predecessors[neighbor.index] = current_node.index
                heapq.heappush(queue, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current_index = goal_node.index
    while current_index is not None:
        path.append(current_index)
        current_index = predecessors[current_index]
    
    return path[::-1]  # Return reversed path

def visualize_prm(nodes, path, environment, robot_type):
    """ Visualize the PRM and the solution path """
    visualize([(node.config, 0) for node in nodes], None, robot_type)  # Visualize nodes only

    # Highlight the solution path
    for i in range(len(path) - 1):
        node1 = nodes[path[i]]
        node2 = nodes[path[i + 1]]
        plt.plot([node1.config[0], node2.config[0]], [node1.config[1], node2.config[1]], 'r-', linewidth=2)
    
    plt.show()

def prm(args):
    """ Main PRM function """
    # Load the environment from the file
    environment = scene_from_file(args.map)
    
    # Step 1: Generate or load configurations
    if not os.path.exists(os.path.join('configs', args.configs)):
        generate_prm_configs(args.robot, 500, args.configs)  # Generate 500 samples if not already generated
    configs = load_configs(args.configs)
    
    # Step 2: Find nearest neighbors for each configuration
    neighbors = nearest_neighbors(args, configs)
    
    # Step 3: Build the PRM graph
    nodes = build_prm(configs, neighbors, environment, args.robot)
    
    # Step 4: Add start and goal nodes
    start_node = Node(args.start, len(nodes))
    goal_node = Node(args.goal, len(nodes) + 1)
    
    nodes.append(start_node)
    nodes.append(goal_node)
    
    # Connect start and goal to nearest neighbors
    start_neighbors = nearest_neighbors(args, configs)
    goal_neighbors = nearest_neighbors(args, configs)
    
    for neighbor in start_neighbors:
        if is_collision_free(start_node.config, neighbor[0], environment, args.robot):
            start_node.edges.append(nodes[configs.index(neighbor[0])])
            nodes[configs.index(neighbor[0])].edges.append(start_node)
    
    for neighbor in goal_neighbors:
        if is_collision_free(goal_node.config, neighbor[0], environment, args.robot):
            goal_node.edges.append(nodes[configs.index(neighbor[0])])
            nodes[configs.index(neighbor[0])].edges.append(goal_node)
    
    # Step 5: Search for the shortest path
    path = dijkstra_search(start_node, goal_node)
    
    # Step 6: Visualize the PRM and the path
    visualize_prm(nodes, path, environment, args.robot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, required=True)  # Either 'freeBody' or 'arm'
    parser.add_argument('--start', type=float, nargs='+', required=True)  # Start configuration
    parser.add_argument('--goal', type=float, nargs='+', required=True)  # Goal configuration
    parser.add_argument('--map', type=str, required=True)  # Environment map file
    parser.add_argument('--configs', type=str, required=True)  # Configurations file

    args = parser.parse_args()
    prm(args)
