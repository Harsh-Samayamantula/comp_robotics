import numpy as np
import argparse
import matplotlib.pyplot as plt
from component_4 import *
import random
import os

def generate_freebody_configs(num_configs, filename):
    os.makedirs('configs', exist_ok=True)
    path = os.path.join('configs', filename)
    with open(path, 'w') as f:
        for _ in range(num_configs):
            x = np.random.uniform(-10, 10) 
            y = np.random.uniform(-10, 10) 
            theta = random.uniform(0, 2 * np.pi)
            f.write(f"{x} {y} {theta}\n")

def generate_arm_configs(num_configs, filename):
    os.makedirs('configs', exist_ok=True)
    path = os.path.join('configs', filename)
    with open(path, 'w') as f:
        for _ in range(num_configs):
            theta0 = random.uniform(0, 2 * np.pi)
            theta1 = random.uniform(0, 2 * np.pi)
            f.write(f"{theta0} {theta1}\n")

def load_configs(filename):
    path = os.path.join('configs', filename)
    with open(path, 'r') as f:
        configs = [list(map(float, line.strip().split())) for line in f]
    return configs

def euclidean_distance(start, end):
    x1, y1, theta1 = start
    x2, y2, theta2 = end
    position_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angle_distance = min(abs(theta2 - theta1), 2 * np.pi - abs(theta2 - theta1))
    return position_distance + angle_distance

def toroidal_distance(config, target):
    theta0_1, theta1_1 = config 
    theta0_2, theta1_2 = target
    
    d_theta0 = min(abs(theta0_2 - theta0_1), 2 * np.pi - abs(theta0_2 - theta0_1))
    d_theta1 = min(abs(theta1_2 - theta1_1), 2 * np.pi - abs(theta1_2 - theta1_1))
    
    return np.sqrt(d_theta0 ** 2 + d_theta1 ** 2)

def nearest_neighbors(args, configs):
    distances = []
    if args.robot == 'freeBody':
        # Compute using Euclidean Distance
        for config in configs:
            distances.append((config, euclidean_distance(config, args.target)))
    else: # robot = 'arm'
        # Compute using angular euclidean function
        for config in configs:
            distances.append((config, toroidal_distance(config, args.target)))
    distances.sort(key=lambda x: x[1])
    return distances[:args.k]

def visualize(distances, target, robot_type):
    fig, ax = plt.subplots()
    
    if robot_type == 'freeBody':
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_aspect('equal')
        ax.grid(True)
        
        target_rect = Rectangle((target[0] - 0.25, target[1] - 0.15), 0.5, 0.3, angle=target[2], color='red')
        ax.add_patch(target_rect)
        
        for config, _ in distances:
            rect = Rectangle((config[0] - 0.25, config[1] - 0.15), 0.5, 0.3, angle=config[2], color='blue', alpha=0.5)
            ax.add_patch(rect)

        plt.title(f'Nearest Neighbors Visualization - Free Body')
    
    elif robot_type == 'arm':
        link1_length, link2_length = 2, 1.5
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal')
        ax.grid(True)

        plot_arm(ax, target, link1_length, link2_length, 'red')

        for config, _ in distances:
            plot_arm(ax, config, link1_length, link2_length, 'blue', alpha=0.5)
        
        plt.title(f'Nearest Neighbors Visualization - Arm')
    
    plt.show()

def plot_arm(ax, config, link1_length, link2_length, color, alpha=1.0):
    theta0, theta1 = config
    J0 = np.array([0, 0])
    J1 = J0 + rotation_matrix(theta0) @ np.array([link1_length, 0])
    J2 = J1 + rotation_matrix(theta0 + theta1) @ np.array([link2_length, 0])
    
   
    ax.plot([J0[0], J1[0]], [J0[1], J1[1]], color=color, lw=4, alpha=alpha)
    ax.plot([J1[0], J2[0]], [J1[1], J2[1]], color=color, lw=4, alpha=alpha)

    ax.plot(J0[0], J0[1], 'o', color=color, alpha=alpha)
    ax.plot(J1[0], J1[1], 'o', color=color, alpha=alpha)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, required=True)
    parser.add_argument('--target', type=float, nargs='+', required=True)
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('--configs', type=str, required=True)

    args = parser.parse_args()
    configs = load_configs(args.configs)
    neighbors = nearest_neighbors(args, configs)
    
    for neighbor in neighbors:
        print(f"Configuration: {neighbor[0]}, Distance: {neighbor[1]}")
    
    visualize(neighbors, args.target, args.robot)

if __name__ == "__main__":
    # generate_freebody_configs(10, 'configs.txt')
    generate_arm_configs(10, 'configs.txt')
    main()

