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

def extend(nearest_node, random_sample, step_size=0.4):
    
    direction = np.array(random_sample) - np.array(nearest_node)
    norm = np.linalg.norm(direction)
    if norm > step_size:
        direction = direction / norm * step_size  # Limit the step size
    new_config = nearest_node + direction
    return new_config


def build_rrt(robot_type, start_config, goal_config, environment, goal_radius=0.1, max_nodes=1000):
    tree = nx.Graph()
    tree.add_node(0, config=start_config)

    i = 1
    while tree.number_of_nodes() < max_nodes:
    
        random_sample = sample_config_rrt(robot_type, goal_config)
        

        configurations = [node['config'] for index, node in tree.nodes(data=True)]
        nearest_node, nearest_conf, _ = nearest_neighbors(robot_type, random_sample, configurations, 1, debug=False)[0]
        
        new_config = extend(nearest_conf, random_sample)
        
        if collision_free_conf(robot_type, new_config, environment, debug=False):
            if is_collision_free((nearest_conf, new_config), environment, robot_type):
                tree.add_node(i, config=new_config)
                tree.add_edge(nearest_node, i)                
                
                if np.linalg.norm(new_config - goal_config) < goal_radius:
                    print(f"Goal reached after {i} nodes.")
                    return tree, i
                i+=1

    
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
    
    if not collision_free_conf(robot_type, start_config, environment, debug=False):
        raise ValueError("Invalid starting configuration for robot")
        
    # Build RRT
    tree, goal_node = build_rrt(robot_type, start_config, goal_config, environment, goal_radius=goal_radius)
    
    # Visualize the tree and solution path if found
    visualize_rrt(tree, start_config, goal_config, environment, goal_radius, robot_type)
    
    if goal_node is not None:
        # Backtrack the path from goal to start
        path = nx.shortest_path(tree, source=0, target=goal_node)
        print('Path found!')
        path_configurations = [tree.nodes[node]['config'] for node in path]
        print(path_configurations)
        animate_solution([tree.nodes[node]['config'] for node in path], robot_type, environment)

def rotation_matrix(theta):
    """Returns the 2D rotation matrix for a given angle theta."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def animate_solution(path, robot_type, environment):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set up the environment (obstacles)
    for obstacle in environment:
        obs_corners = get_corners(obstacle['position'], obstacle['width'], obstacle['height'], obstacle['orientation'])
        obs = plt.Polygon(obs_corners, edgecolor='black', facecolor='green')
        ax.add_patch(obs)

    # Plot the full static path
    if robot_type == "arm":
        path_x, path_y = zip(*[(np.cos(p[0]) + np.cos(p[0] + p[1]), np.sin(p[0]) + np.sin(p[0] + p[1])) for p in path])
    else:  # freeBody robot
        path_x, path_y = zip(*[(p[0], p[1]) for p in path])
    
    ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')

    # Set plot limits
    if robot_type == "arm":
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    else:  # freeBody
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    ax.set_aspect('equal')
    ax.grid(True)

    # Initialize plot elements
    if robot_type == "arm":
        # Two lines for the arm links and a point for the end effector
        line1, = ax.plot([], [], 'ro-', lw=4)  # First link
        line2, = ax.plot([], [], 'bo-', lw=4)  # Second link
        end_effector, = ax.plot([], [], 'ko', markersize=8)  # End effector
    else:  # freeBody
        # Initialize a rectangle for the robot's body
        body_rect = plt.Rectangle((0, 0), 0.5, 0.3, angle=0, color='blue', alpha=0.5)
        ax.add_patch(body_rect)

    # Function to initialize the animation
    def init():
        if robot_type == "arm":
            line1.set_data([], [])
            line2.set_data([], [])
            end_effector.set_data([], [])
            return line1, line2, end_effector
        else:
            body_rect.set_xy((0, 0))  # Reset the rectangle position
            body_rect.set_angle(0)  # Reset orientation
            return body_rect,

    # Function to update the frame
    def update(frame):
        if robot_type == "arm":
            theta1, theta2 = path[frame]  # Get the joint angles from the path

            # Compute the position of each link's end
            base = np.array([0, 0])
            joint1 = base + rotation_matrix(theta1) @ np.array([1.5, 0])  # First link length
            end_eff = joint1 + rotation_matrix(theta1 + theta2) @ np.array([1.0, 0])  # Second link length

            # Update the line data for the arm
            line1.set_data([base[0], joint1[0]], [base[1], joint1[1]])
            line2.set_data([joint1[0], end_eff[0]], [joint1[1], end_eff[1]])
            end_effector.set_data(end_eff[0], end_eff[1])

            return line1, line2, end_effector
        else:
            x, y, theta = path[frame]  # Get position and orientation from the path

            # Update the rectangle position and angle
            body_rect.set_xy((x + 0.25, y - 0.15))  # Adjust rectangle center
            body_rect.set_angle(np.degrees(theta))  # Update angle

            return body_rect,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, repeat=False)

    plt.title(f"Animating {robot_type} Robot Traversing the Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

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
