import os
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
                         obstacle['width'], obstacle['height'], 
                         edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Draw robot
    robot_rect = Rectangle((robot['position'][0] - robot['width']/2, 
                            robot['position'][1] - robot['height']/2), 
                           robot['width'], robot['height'], 
                           edgecolor='blue', facecolor='none')
    ax.add_patch(robot_rect)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
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
        robot_position = (random.uniform(-10, 10), random.uniform(-10, 10))
        robot = {'position': robot_position, 'width': robot_size[0], 'height': robot_size[1]}
        
        # Check for collisions with all obstacles
        colliding_indices = []
        for i, obstacle in enumerate(environment):
            if check_collision(robot, obstacle):
                colliding_indices.append(i)
        
        # Visualize the environment with collision indication
        visualize_scene_with_collisions(environment, robot, colliding_indices)

if __name__ == '__main__':
    # Run collision checking with the specified environment file
    collision_checking('environment_4.txt')  # Modify the filename as needed