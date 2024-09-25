import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from utils import *
from component_3 import *
from component_2 import *
from component_1 import *

# Ensure theta is in range [0, 2pi]
# Ensure check SEn and check SOn work for n=2
# Ensure determinant correction works for correct methods

# Helper function to create a 2D rotation matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

# Interpolate arm configurations between start and goal
def interpolate_arm(start, goal):
    theta0_0, theta1_0 = start
    theta0_1, theta1_1 = goal
    
    # Create a list of interpolated poses
    robot_path = []
    steps = 10
    for t in np.linspace(0, 1, steps):
        theta0 = (1 - t) * theta0_0 + t * theta0_1
        theta1 = (1 - t) * theta1_0 + t * theta1_1
        robot_path.append([theta0, theta1])
    
    return np.array(robot_path)

# Forward propagate the arm given a plan
def forward_propagate_arm(start_pose, plan):
    theta0, theta1 = start_pose
    robot_path = [start_pose]
    
    for velocity, duration in plan:
        v0, v1 = velocity
        theta0 += v0 * duration
        theta1 += v1 * duration
        robot_path.append([theta0, theta1])
    
    return np.array(robot_path)

# Visualize the arm's movement
def visualize_arm_path(path):
    link1_length, link2_length = 2, 1.5
    fig, ax = plt.subplots()
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_aspect('equal')
    ax.grid()

    # Function to calculate the link positions based on joint angles
    def calculate_link_positions(theta0, theta1):
        # Base of the first link (fixed at origin)
        J0 = np.array([0, 0])
        
        # End of the first link
        J1 = J0 + rotation_matrix(theta0) @ np.array([link1_length, 0])
        
        # End of the second link
        J2 = J1 + rotation_matrix(theta0 + theta1) @ np.array([link2_length, 0])
        
        return J0, J1, J2

    # Plotting the arm initially
    line, = ax.plot([], [], 'o-', lw=4)
    
    # List to store the end-effector positions
    end_effector_path = []

    def update(frame):
        theta0, theta1 = path[frame]
        J0, J1, J2 = calculate_link_positions(theta0, theta1)
        
        # Update the line to reflect new joint positions
        line.set_data([J0[0], J1[0], J2[0]], [J0[1], J1[1], J2[1]])
        
        # Append the end-effector position to the list
        end_effector_path.append(J2)
        
        # Plot the end-effector path
        ax.plot([p[0] for p in end_effector_path], [p[1] for p in end_effector_path], 'r-', lw=2, label="End-Effector Path")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=100, blit=False, repeat=False)
    
    plt.show()

if __name__ == '__main__':
    # Example usage
    start_pose = (0, 0)  # Initial joint angles (theta0, theta1)
    goal_pose = (np.pi / 4, np.pi / 2)  # Goal joint angles (theta0, theta1)
    example_path = interpolate_arm(start_pose, goal_pose)
    visualize_arm_path(example_path)

    # Example for forward propagation with a plan
    example_plan = [
        ((0.1, 0.05), 2),  # (velocity in theta0, velocity in theta1), duration
        ((-0.1, 0.1), 2),
    ]
    print(forward_propagate_arm(start_pose, example_plan))