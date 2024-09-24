import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import scipy.stats
from utils import *


def interpolate_rigid_body(start_pose, goal_pose):
    x0, y0, theta0 = start_pose
    x1, y1, theta1 = goal_pose

    robot_path = []
    steps = 10
    for t in np.linspace(0, 1, steps):

        # Interpolation done with weighted average
        x = (1-t) * x0 + t * x1
        y = (1-t) * y0 + t * y1
        theta = (1-t) * theta0 + t * theta1

        robot_path.append([x, y, theta])
    
    return np.array(robot_path)



def forward_propogate_rigid_body(start_pose, plan):
    x0, y0, theta0 = start_pose
    robot_path = [start_pose]

    for velocity, duration in plan:
        v_x, v_y, v_theta = velocity

        # Updating pose with the velocity

        x0 += v_x * duration
        y0 += v_y * duration
        theta0 += v_theta * duration

        robot_path.append([x0, y0, theta0])

    return np.array(robot_path)

def visualize_path(path):
    r_width, r_length = 0.5, 0.3

    fig, ax = plt.subplots()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()

    x_coords = [pose[0] for pose in path]
    y_coords = [pose[1] for pose in path]
    theta_coords = [pose[2] for pose in path]
    
    ax.plot(x_coords, y_coords, '-o', label="Path")
    
    robot = Rectangle((x_coords[0] - r_width/2, y_coords[0] - r_length/2), 
                      r_width, r_length, angle=np.degrees(theta_coords[0]), color='blue', alpha=0.5)
    ax.add_patch(robot)

    def update(frame):
        x, y, theta = path[frame]
        
        robot.set_xy((x - r_width / 2, y - r_length / 2))
        robot.angle = np.degrees(theta)
        
        return robot,
    
    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=200, blit=True, repeat=False)

    plt.legend()
    plt.show()
    print('showing now.')

example_path = [
    [-5, -5, 0],
    [-3, -4, np.pi/8],
    [-1, -2, np.pi/4],
    [1, 0, np.pi/2],
    [3, 2, 3*np.pi/4],
    [5, 5, np.pi]
]

visualize_path(example_path)