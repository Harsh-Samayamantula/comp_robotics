import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import networkx as nx
from component_5 import scene_from_file
from component_7 import *

# Define car parameters
L = 2.0  # Length of the car (wheelbase)

# Car dynamics equations
def car_dynamics(state, V, delta, dt=0.1):
    x, y, theta = state
    beta = np.arctan(0.5 * np.tan(delta))
    
    # Update car's state based on its dynamics
    x_dot = V * np.cos(theta + beta)
    y_dot = V * np.sin(theta + beta)
    theta_dot = (2 * V / L) * np.sin(beta)
    
    # New state after applying control for time dt
    new_x = x + x_dot * dt
    new_y = y + y_dot * dt
    new_theta = theta + theta_dot * dt
    
    return np.array([new_x, new_y, new_theta])

# Sample random configuration for the car
def sample_config_car(goal_config=None, goal_bias=0.01):
    if random.random() < goal_bias and goal_config is not None:
        return goal_config
    else:
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        theta = random.uniform(-np.pi, np.pi)  # Orientation can range from -pi to pi
        return np.array([x, y, theta])

def animate_solution(path, environment):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the environment (obstacles)
    for obstacle in environment:
        obs_corners = get_corners(obstacle['position'], obstacle['width'], obstacle['height'], obstacle['orientation'])
        obs = plt.Polygon(obs_corners, edgecolor='black', facecolor='green')
        ax.add_patch(obs)

    # Set plot limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    
    # Draw the path as a line
    path_x, path_y = zip(*[(state[0], state[1]) for state in path])
    ax.plot(path_x, path_y, 'r-', linewidth=2, label="Path")
    
    # Initialize the car body (a rectangle) to represent the car
    car_width = 1.0
    car_height = 0.5
    car_body = plt.Rectangle((0, 0), car_width, car_height, angle=0, color='blue', alpha=0.7)
    ax.add_patch(car_body)

    # Function to update the car's position and orientation for each frame
    def update(frame):
        state = path[frame]
        x, y, theta = state
        
        # Update car's position and orientation
        car_body.set_xy([x - car_width / 2, y - car_height / 2])  # Set position
        car_body.angle = np.degrees(theta)  # Set orientation

        return car_body,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(path), interval=100, blit=True, repeat=False)

    # Show the animation
    plt.show()

# Extend function to generate new states
def extend_car(nearest_node, random_sample, step_size=0.5):
    V = random.uniform(0, 1)  # Random velocity
    delta = random.uniform(-np.pi / 3, np.pi / 3)  # Steering angle
    
    # Apply car dynamics to move from nearest_node towards random_sample
    new_state = car_dynamics(nearest_node, V, delta, dt=step_size)
    
    return new_state

# Build RRT for car-like robot
def build_rrt_car(start_config, goal_config, environment, goal_radius=0.5, max_nodes=1000):
    tree = nx.Graph()
    tree.add_node(0, config=start_config)
    i = 1
    
    while tree.number_of_nodes() < max_nodes:
        random_sample = sample_config_car(goal_config)
        # x = random.uniform(-20, 0)
        # y = random.uniform(-20, 0)
        # theta = random.uniform(-np.pi, np.pi)
        # random_sample = np.array([x, y, theta])
        configurations = [node['config'] for _, node in tree.nodes(data=True)]
        
        # Find nearest node
        nearest_node, nearest_conf, _ = nearest_neighbors("freeBody", random_sample, configurations, 1)[0]
        
        # Extend the tree in the direction of the random_sample
        new_config = extend_car(nearest_conf, random_sample)
        
        # Check for collisions (function not provided, assuming a pre-defined collision checker)
        if collision_free_conf("freeBody", new_config, environment):
            if is_collision_free((nearest_conf, new_config), environment, "freeBody"):
                tree.add_node(i, config=new_config)
                tree.add_edge(nearest_node, i)
                
                # Check if goal is reached
                if np.linalg.norm(new_config[:2] - goal_config[:2]) < goal_radius:
                    print(f"Goal reached after {i} nodes.")
                    return tree, i
                i += 1
    
    print("Max nodes reached without finding the goal.")
    return tree, None

# Visualize RRT and trajectory for the car
def visualize_rrt_car(tree, start_config, goal_config, environment, goal_radius=0.5):
    plt.figure(figsize=(10, 10))
    
    # Obstacles
    for obstacle in environment:
        obs_corners = get_corners(obstacle['position'], obstacle['width'], obstacle['height'], obstacle['orientation'])
        obs = plt.Polygon(obs_corners, edgecolor='black', facecolor='green')
        plt.gca().add_patch(obs)
    
    # Plot the tree
    for node1, node2 in tree.edges:
        config1 = tree.nodes[node1]['config']
        config2 = tree.nodes[node2]['config']
        plt.plot([config1[0], config2[0]], [config1[1], config2[1]], 'b-')
    
    # Start and goal
    plt.scatter(start_config[0], start_config[1], c='g', marker='o', s=100, label="Start")
    plt.scatter(goal_config[0], goal_config[1], c='r', marker='x', s=100, label="Goal")
    plt.gca().add_patch(plt.Circle(goal_config[:2], goal_radius, color='red', fill=False, linestyle='--', label="Goal Region"))
    
    plt.title("RRT for Car-like Robot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Main function to execute RRT for car-like robot
def main(start_config, goal_config, map_file, goal_radius=0.5):
    environment = scene_from_file(map_file)
    
    if not collision_free_conf("freeBody", start_config, environment):
        raise ValueError("Invalid start configuration for car")
    
    tree, goal_node = build_rrt_car(start_config, goal_config, environment, goal_radius=goal_radius)
    
    visualize_rrt_car(tree, start_config, goal_config, environment, goal_radius)
    if goal_node is not None:
        path = nx.shortest_path(tree, source=0, target=goal_node)
        path_configurations = [tree.nodes[node]['config'] for node in path]
        print("Path found:", path_configurations)

        # Visualize the car driving through the path
        animate_solution(path_configurations, environment)
    else:
        print("No valid path found.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RRT for Car-like Robot")
    parser.add_argument("--start", type=float, nargs=3, required=True, help="Start configuration (x, y, theta)")
    parser.add_argument("--goal", type=float, nargs=3, required=True, help="Goal configuration (x, y, theta)")
    parser.add_argument("--map", type=str, required=True, help="Map file")
    parser.add_argument("--goal_rad", type=float, default=0.5, help="Goal radius")
    
    args = parser.parse_args()
    
    main(np.array(args.start), np.array(args.goal), args.map, args.goal_rad)