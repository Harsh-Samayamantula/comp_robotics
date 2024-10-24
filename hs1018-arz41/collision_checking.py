import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle
from component_5 import scene_from_file

def check_collision(robot, obstacle):
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

def is_collision_free(path, environment, robot_type):
    """
    Check if the path between two configurations is collision-free.
    Path is a tuple of two configurations (start, goal).
    """
    start, goal = path
    
    if robot_type == 'freeBody':
        # Interpolate between the start and goal for rigid body
        robot_path = interpolate_rigid_body(start, goal)
    elif robot_type == 'arm':
        # Interpolate between the start and goal for arm
        robot_path = interpolate_arm(start, goal)
        
    
    # Check if all configurations along the path are collision-free
    # print(f'ROBOT PATH {start} {goal}')
    # print(robot_path)
    for config in robot_path:
        # print(f'Checking {config}')
        if not collision_free_conf(robot_type, config, environment):
            return False
    
    return True

def visualize_scene_with_collisions(environment, robot, colliding_indices):
    """
    Visualize the environment and robot. Colliding obstacles are red, non-colliding obstacles are green.
    """
    fig, ax = plt.subplots()
    
    # Draw obstacles and color them based on collision status
    for i, obstacle in enumerate(environment):
        color = 'red' if i in colliding_indices else 'green'
        rect = Rectangle((obstacle['position'][0] - obstacle['width']/2, 
                          obstacle['position'][1] - obstacle['height']/2), 
                         obstacle['width'], obstacle['height'], angle=np.degrees(obstacle['orientation']),
                         edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Draw robot
    robot_rect = Rectangle((robot['position'][0] - robot['width']/2, 
                            robot['position'][1] - robot['height']/2), 
                           robot['width'], robot['height'], angle=np.degrees(robot['orientation']),
                           edgecolor='blue', facecolor='none')
    ax.add_patch(robot_rect)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    plt.show()

def collision_checking(environment_file):
    """
    Perform collision checking by placing the robot randomly for 10 seconds and checking for collisions.
    """
    # Load the environment
    environment = scene_from_file(environment_file)
    
    robot_size = (0.5, 0.3)  # Robot dimensions (width, height)
    
    # Perform 10 random poses of the robot (1 pose per second)
    for _ in range(10):
        # Random pose for the robot
        robot_position = (random.uniform(0, 20), random.uniform(0, 20))
        robot_theta = random.uniform(0, 2 * np.pi)
        robot = {'position': robot_position, 'width': robot_size[0], 'height': robot_size[1], 'orientation': robot_theta}
        
        # Check for collisions with all obstacles
        colliding_indices = []
        for i, obstacle in enumerate(environment):
            if check_collision(robot, obstacle):
                colliding_indices.append(i)
        
        # Visualize the environment with collision indication
        visualize_scene_with_collisions(environment, robot, colliding_indices)

def main():
    parser = argparse.ArgumentParser(description="Process a map argument.")
    
    # Adding the --map argument
    parser.add_argument('--map', type=str, help='Path to the map file or description')

    # Adding the --robot argument
    parser.add_argument('--robot', type=str, help='Robot type (arm or free body)')

    # Parse the arguments
    args = parser.parse_args()
    
    # Run collision checking with the specified environment file
    if args.map:
        collision_checking(args.map)
    else:
        print("No map argument provided. Running on \"environment_4.txt\"")

if __name__ == '__main__':
    main()