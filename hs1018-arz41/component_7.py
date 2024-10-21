import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time

# Define obstacle class
class Obstacle:
    def __init__(self, center, width, height, angle):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle  # Rotation angle in radians

# Generate a random environment with rectangular obstacles
def load_environment(file_name):
    obstacles = []
    with open(file_name, 'r') as file:
        for line in file:
            data = line.strip().split()
            center = (float(data[0]), float(data[1]))
            width = float(data[2])
            height = float(data[3])
            angle = float(data[4])
            obstacles.append(Obstacle(center, width, height, angle))
    return obstacles

# Check if two rectangles collide (simplified for axis-aligned case)
def check_collision(robot, obstacle):
    rx, ry = robot['center']
    rw, rh = robot['size']
    
    ox, oy = obstacle.center
    ow, oh = obstacle.width, obstacle.height
    
    # Check if rectangles overlap (simple AABB)
    if (rx < ox + ow/2 and rx + rw/2 > ox - ow/2 and 
        ry < oy + oh/2 and ry + rh/2 > oy - oh/2):
        return True
    return False

# Visualize the environment and robot
def visualize_environment(obstacles, robot, colliding_indices):
    fig, ax = plt.subplots()
    
    # Draw obstacles
    for i, obstacle in enumerate(obstacles):
        color = 'red' if i in colliding_indices else 'green'
        rect = patches.Rectangle((obstacle.center[0] - obstacle.width/2, 
                                  obstacle.center[1] - obstacle.height/2), 
                                 obstacle.width, obstacle.height, 
                                 angle=np.degrees(obstacle.angle), 
                                 edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Draw robot
    robot_rect = patches.Rectangle((robot['center'][0] - robot['size'][0]/2, 
                                    robot['center'][1] - robot['size'][1]/2), 
                                   robot['size'][0], robot['size'][1], 
                                   edgecolor='blue', facecolor='none')
    ax.add_patch(robot_rect)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.show()

# Main function for collision checking
def collision_checking(environment_file):
    # Load the environment
    obstacles = load_environment(environment_file)
    
    robot_size = (0.5, 0.3)  # Robot dimensions (width, height)
    
    # Perform 10 random poses of the robot
    for _ in range(10):
        # Random pose for the robot
        robot_center = (random.uniform(0, 20), random.uniform(0, 20))
        robot = {'center': robot_center, 'size': robot_size}
        
        # Check for collisions with all obstacles
        colliding_indices = []
        for i, obstacle in enumerate(obstacles):
            if check_collision(robot, obstacle):
                colliding_indices.append(i)
        
        # Visualize the result
        visualize_environment(obstacles, robot, colliding_indices)
        
        time.sleep(1)  # Pause for 1 second before the next pose

# Example usage:
# Run the collision checking on a provided map file
if __name__ == '__main__':
    collision_checking('map.txt')