import os
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_environment(number_of_obstacles):
    environment = []
    for _ in range(number_of_obstacles):
        w, h = random.uniform(0.5, 2), random.uniform(0.5, 2)
        x, y = random.uniform(2, 18), random.uniform(2, 18)
        theta = random.uniform(0, 2 * np.pi)
        obstacle = {'width': w, 'height': h, 'position': (x, y), 'orientation': theta}
        environment.append(obstacle)
    return environment

def scene_to_file(environment, filename):
    os.makedirs('env_files', exist_ok=True)
    path = os.path.join('env_files', filename)
    with open(path, 'w') as f:
        for obstacle in environment:
            f.write(f"{obstacle['width']} {obstacle['height']} {obstacle['position'][0]} {obstacle['position'][1]} {obstacle['orientation']}\n")

def scene_from_file(filename):
    environment = []
    path = os.path.join('env_files', filename)
    with open(path, 'r') as f:
        for line in f:
            w, h, x, y, theta = map(float, line.strip().split())
            environment.append({'width': w, 'height': h, 'position': (x, y), 'orientation': theta})
    return environment

def visualize_scene(environment):
    fig, ax = plt.subplots()
    for obstacle in environment:
        rect = plt.Rectangle((obstacle['position'][0], obstacle['position'][1]), obstacle['width'], obstacle['height'], angle=np.degrees(obstacle['orientation']), edgecolor='black', facecolor='green')
        ax.add_patch(rect)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    envs = [3, 6, 9, 12, 15]
    for env_num, obs_num in enumerate(envs):
        env = generate_environment(obs_num)
        scene_to_file(env, 'environment_' + str(env_num) + '.txt')
        visualize_scene(env)
    

    
