import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from component_5 import *
from component_6 import *  # Import the nearest_neighbors function
from component_7 import *
import networkx as nx

def sample_config(robot_type):
    if robot_type == "arm":
        return np.array([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])
    elif robot_type == "freeBody":
        return np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), random.uniform(0, 2 * np.pi)])
    else:
        raise ValueError("Invalid robot type")


def build_prm(robot_type, environment, n_samples=1000, k=6, debug=False):
    graph = nx.Graph()
    samples = []
    
    for _ in range(n_samples):
        config = sample_config(robot_type)
        # if is_collision_free(config, environment, robot_type):
        if debug: print('Config Passed In:', config)
        if collision_free_conf(robot_type, config, environment, debug=debug):
            samples.append(config)
            graph.add_node(len(samples) - 1, config=config)
            if debug: print('Samples Passed In:', len(samples))
            
    
    for i, sample in enumerate(samples):
        # Use the nearest_neighbors function from component_6.py
        neighbors = nearest_neighbors(robot_type, sample, samples, k)
        for index, conf, dist in neighbors:
            if not graph.has_edge(i, index) and i != index:
                if is_collision_free((sample, samples[index]), environment, robot_type):
                    graph.add_edge(i, index, weight=dist)
    # if debug: print('Len of Samples', len(samples))
    return graph, samples

def find_path(graph, start_config, goal_config, robot_type, samples):
    # start_node = min(range(len(samples)), key=lambda i: distance(samples[i], start_config, robot_type))
    s_node, s_conf, s_dist = nearest_neighbors(robot_type, start_config, samples, 1, debug=False)[0]
    g_node, g_conf, g_dist = nearest_neighbors(robot_type, goal_config, samples, 1)[0]
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
        obs_corners = get_corners(obstacle['position'], obstacle['width'], obstacle['height'], obstacle['orientation'])
        obs = plt.Polygon(obs_corners, edgecolor='black', facecolor='green')
        plt.gca().add_patch(obs)

    if robot_type == "arm":
        # Function to calculate the link positions based on joint angles
        def calculate_link_positions(theta0, theta1):
            # Base of the first link (fixed at origin)
            J0 = np.array([0, 0])

            # End of the first link
            J1 = J0 + rotation_matrix(theta0) @ np.array([2.0, 0])  # Adjust length of link 1

            # End of the second link
            J2 = J1 + rotation_matrix(theta0 + theta1) @ np.array([1.5, 0])  # Adjust length of link 2

            return J0, J1, J2

        # Visualize the PRM graph with arm configurations (not just points)
        for node in samples:
            theta0, theta1 = node
            J0, J1, J2 = calculate_link_positions(theta0, theta1)

            plt.plot([J0[0], J1[0], J2[0]], [J0[1], J1[1], J2[1]], 'bo-', lw=1)

        # Plot path if available
        if path:
            for node in path:
                theta0, theta1 = node
                J0, J1, J2 = calculate_link_positions(theta0, theta1)
                plt.plot([J0[0], J1[0], J2[0]], [J0[1], J1[1], J2[1]], 'ro-', lw=2)
    else:    
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
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_aspect('equal')
    ax.grid(True)

    # Initialize plot elements
    if robot_type == "arm":
        # Two lines for the arm links and a point for the end effector
        line1, = ax.plot([], [], 'r-', lw=4)  # First link
        line2, = ax.plot([], [], 'b-', lw=4)  # Second link
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

def main(robot_type, start_config, goal_config, map_file):
    environment = scene_from_file(map_file)
    
    graph, samples = build_prm(robot_type, environment, debug=False)
    path = find_path(graph, start_config, goal_config, robot_type, samples)
    
    if path:
        print("Path found!")
        print(path)
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