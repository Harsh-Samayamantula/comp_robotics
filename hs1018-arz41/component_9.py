import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle, Circle
from collections import namedtuple
import math

# Named tuple for a Node in the RRT
Node = namedtuple('Node', ['config', 'parent'])

def euclidean_distance(q1, q2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(q1) - np.array(q2))

def steer(q_nearest, q_rand, step_size):
    """Steer from nearest node towards the random node within a step size."""
    direction = np.array(q_rand) - np.array(q_nearest)
    length = np.linalg.norm(direction)
    direction = direction / length  # Normalize the direction
    return tuple(np.array(q_nearest) + step_size * direction)

def random_config(robot_type, bounds):
    """Generate a random configuration in the robot's configuration space."""
    if robot_type == 'freeBody':
        return (random.uniform(bounds[0][0], bounds[0][1]),
                random.uniform(bounds[1][0], bounds[1][1]),
                random.uniform(-np.pi, np.pi))  # Random pose (x, y, theta)
    elif robot_type == 'arm':
        return (random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi))  # Random angles for arm

def collision_check(robot, obstacle):
    """
    Check for collision between the robot and an obstacle.
    We are using simple axis-aligned bounding box (AABB) logic here for simplicity.
    """
    rx, ry = robot['position']
    rw, rh = robot['width'], robot['height']
    
    ox, oy = obstacle['position']
    ow, oh = obstacle['width'], obstacle['height']
    
    # Check if the robot's bounding box overlaps with the obstacle's bounding box
    if (rx < ox + ow/2 and rx + rw/2 > ox - ow/2 and 
        ry < oy + oh/2 and ry + rh/2 > oy - oh/2):
        return True
    return False

def load_obstacles(map_file):
    """
    Load obstacles from the environment file.
    The file is expected to contain rectangular obstacles defined by position, width, and height.
    """
    obstacles = []
    with open(map_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            obstacle = {'position': (parts[0], parts[1]), 'width': parts[2], 'height': parts[3]}
            obstacles.append(obstacle)
    return obstacles

def nearest_node(tree, q_rand):
    """Find the nearest node in the tree to the random configuration."""
    min_dist = float('inf')
    nearest = None
    for node in tree:
        dist = euclidean_distance(node.config, q_rand)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest

def is_goal_reached(node, goal, goal_radius):
    """Check if the current node is within the goal radius."""
    return euclidean_distance(node.config, goal) < goal_radius

def rrt(robot_type, start, goal, goal_radius, map_file, step_size=0.2, max_iterations=1000, goal_sample_rate=0.05):
    """
    Run the RRT algorithm to find a path from start to goal while avoiding obstacles.
    """
    obstacles = load_obstacles(map_file)
    bounds = [(-10, 10), (-10, 10)]  # Bounds for the freeBody robot's x and y space

    # Initialize the tree with the start configuration
    tree = [Node(config=start, parent=None)]
    
    for iteration in range(max_iterations):
        # Sample random configuration with a probability of sampling the goal
        if random.random() < goal_sample_rate:
            q_rand = goal
        else:
            q_rand = random_config(robot_type, bounds)
        
        # Find nearest node in the tree to q_rand
        q_nearest = nearest_node(tree, q_rand)

        # Steer towards q_rand, within step size
        q_new = steer(q_nearest.config, q_rand, step_size)

        # Check for collisions before adding the new node
        robot = {'position': q_new[:2], 'width': 0.5, 'height': 0.3}
        if any(check_collision(robot, obs) for obs in obstacles):
            continue  # Collision detected, skip this node

        # Add new node to the tree
        tree.append(Node(config=q_new, parent=q_nearest))

        # Check if goal has been reached
        if is_goal_reached(tree[-1], goal, goal_radius):
            return tree, tree[-1]  # Return the tree and the goal node

    # If no path is found, return the tree and None
    return tree, None

def visualize_rrt(tree, goal_node, goal, robot_type, obstacles):
    """
    Visualize the RRT tree in the configuration space and the path to the goal.
    """
    fig, ax = plt.subplots()
    
    # Draw obstacles
    for obs in obstacles:
        rect = Rectangle((obs['position'][0] - obs['width']/2, obs['position'][1] - obs['height']/2),
                         obs['width'], obs['height'], edgecolor='black', facecolor='gray')
        ax.add_patch(rect)
    
    # Draw the RRT tree
    for node in tree:
        if node.parent:
            ax.plot([node.config[0], node.parent.config[0]],
                    [node.config[1], node.parent.config[1]], 'b-')

    # Draw the path to the goal if it exists
    if goal_node:
        path_node = goal_node
        while path_node.parent:
            ax.plot([path_node.config[0], path_node.parent.config[0]],
                    [path_node.config[1], path_node.parent.config[1]], 'r-', linewidth=2)
            path_node = path_node.parent
    
    # Draw the goal
    goal_circle = Circle(goal[:2], radius=goal[2], edgecolor='green', facecolor='none')
    ax.add_patch(goal_circle)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)
    
    args = parser.parse_args()

    # Run the RRT algorithm
    tree, goal_node = rrt(
        robot_type=args.robot,
        start=tuple(args.start),
        goal=tuple(args.goal),
        goal_radius=args.goal_rad,
        map_file=args.map
    )

    # Visualize the result
    obstacles = load_obstacles(args.map)
    visualize_rrt(tree, goal_node, tuple(args.goal) + (args.goal_rad,), args.robot, obstacles)

if __name__ == '__main__':
    main()
