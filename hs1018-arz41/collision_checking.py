import numpy as np
import matplotlib.pyplot as plt
import argparse
from environment import scene_from_file

def check_collision(robot, obstacle):
    rx, ry = robot['position']
    rw, rh = robot['width'], robot['height']
    
    ox, oy = obstacle['position']
    ow, oh = obstacle['width'], obstacle['height']
    
    if (rx < ox + ow/2 and rx + rw/2 > ox - ow/2 and 
        ry < oy + oh/2 and ry + rh/2 > oy - oh/2):
        return True
    return False

def visualize_scene_with_collisions(environment, robot, colliding_indices):
    fig, ax = plt.subplots()
    
    for i, obstacle in enumerate(environment):
        color = 'red' if i in colliding_indices else 'green'
        rect = plt.Rectangle((obstacle['position'][0] - obstacle['width']/2, 
                              obstacle['position'][1] - obstacle['height']/2), 
                             obstacle['width'], obstacle['height'], 
                             angle=np.degrees(obstacle['orientation']),
                             edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    robot_rect = plt.Rectangle((robot['position'][0] - robot['width']/2, 
                                robot['position'][1] - robot['height']/2), 
                               robot['width'], robot['height'], 
                               edgecolor='blue', facecolor='none')
    ax.add_patch(robot_rect)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    plt.show()

def collision_checking(robot_type, map_file):
    environment = scene_from_file(map_file)
    
    if robot_type == "arm":
        robot_size = (0.5, 0.3)
    elif robot_type == "freeBody":
        robot_size = (0.5, 0.5)
    
    for _ in range(10):
        robot_position = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        robot = {'position': robot_position, 'width': robot_size[0], 'height': robot_size[1]}
        
        colliding_indices = []
        for i, obstacle in enumerate(environment):
            if check_collision(robot, obstacle):
                colliding_indices.append(i)
        
        visualize_scene_with_collisions(environment, robot, colliding_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision Checking")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Robot type")
    parser.add_argument("--map", type=str, required=True, help="Map file")
    
    args = parser.parse_args()
    
    collision_checking(args.robot, args.map)