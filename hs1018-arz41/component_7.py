import os
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Polygon, Rectangle
from component_5 import *
from component_4 import *
from component_3 import *
import math

def rotate_point(point, angle):
    """
    Rotate a point (x, y) by a given angle around the origin.
    """
    x, y = point
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle)

def get_corners(position, width, height, orientation):
    """
    Get the four corners of the robot given its position, width, height, and orientation.
    """
    cx, cy = position
    half_w, half_h = width / 2, height / 2
    
    # The corners of the rectangle before rotation
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    
    # Rotate each corner by the orientation angle and then translate it to the robot's position
    rotated_corners = [rotate_point(corner, orientation) for corner in corners]
    translated_corners = [(cx + x, cy + y) for x, y in rotated_corners]
    
    return translated_corners

def check_collision(robot, obstacle, arm=False):
    """
    Check for collision between the robot and an obstacle, taking into account the robot's orientation.
    The robot is treated as a rotated rectangle and the obstacle as an axis-aligned bounding box (AABB).
    """
    robot_corners = ''
    if not arm: robot_corners = get_corners((robot[0], robot[1]), 0.5, 0.3, robot[2])
    else: robot_corners = get_corners((robot['position'][0], robot['position'][1]), robot['width'], robot['height'], robot['orientation'])
    
    ox, oy = obstacle['position']
    ow, oh = obstacle['width'], obstacle['height']
    
    # Check for collision using the Separating Axis Theorem (SAT) for rotated rectangles and AABB
    obstacle_corners = [(ox - ow/2, oy - oh/2), (ox + ow/2, oy - oh/2), (ox + ow/2, oy + oh/2), (ox - ow/2, oy + oh/2)]
    
    return polygons_collide(robot_corners, obstacle_corners)

def project_polygon(axis, polygon):
    """
    Project a polygon onto an axis and return the min and max values of the projection.
    """
    projections = [np.dot(axis, corner) for corner in polygon]
    return min(projections), max(projections)

def polygons_collide(polygon1, polygon2):
    """
    Use the Separating Axis Theorem (SAT) to check if two polygons collide.
    """
    # Get all edges from both polygons
    edges = []
    for i in range(len(polygon1)):
        p1 = polygon1[i]
        p2 = polygon1[(i + 1) % len(polygon1)]
        edge = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        edges.append(edge)
    
    for i in range(len(polygon2)):
        p1 = polygon2[i]
        p2 = polygon2[(i + 1) % len(polygon2)]
        edge = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        edges.append(edge)
    
    # For each edge, compute the normal (perpendicular) axis and project both polygons onto that axis
    for edge in edges:
        axis = np.array([-edge[1], edge[0]])  # Perpendicular axis
        axis = axis / np.linalg.norm(axis)    # Normalize the axis
        
        proj1_min, proj1_max = project_polygon(axis, polygon1)
        proj2_min, proj2_max = project_polygon(axis, polygon2)
        
        # If there is no overlap in the projections, the polygons do not collide
        if proj1_max < proj2_min or proj2_max < proj1_min:
            return False
    
    # If all projections overlap, the polygons collide
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
                         obstacle['width'], obstacle['height'], 
                         edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Draw robot
    robot_corners = get_corners(robot['position'], robot['width'], robot['height'], robot['orientation'])
    polygon = Polygon(robot_corners, edgecolor='blue', facecolor='none')
    ax.add_patch(polygon)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    plt.show()

def forward_kinematics(theta0, theta1):
    """
    Compute the positions of the two links based on joint angles.
    Link1 has length 2, and Link2 has length 1.5.
    """
    link1_length = 2
    link2_length = 1.5
    
    # Base of the arm is at the origin (0, 0)
    x0, y0 = 0, 0
    
    # Compute the position of the end of the first link
    x1 = x0 + link1_length * np.cos(theta0)
    y1 = y0 + link1_length * np.sin(theta0)
    
    # Compute the position of the end of the second link
    x2 = x1 + link2_length * np.cos(theta0 + theta1)
    y2 = y1 + link2_length * np.sin(theta0 + theta1)
    
    return [(x0, y0), (x1, y1), (x2, y2)]  # Base, end of link1, end of link2

def get_link_boxes(arm_positions, theta0, theta1):
    """
    Generate rectangles representing the links of the arm, given the positions of the joints and end-effectors.
    Each link is represented as a box centered on the line segment between its two endpoints, with orientation.
    """
    link1_length = 2
    link2_length = 1.5
    link_width = 0.2  # Assume a fixed width for both links

    # Link1 is between the base (0,0) and the first joint
    x0, y0 = arm_positions[0]
    x1, y1 = arm_positions[1]

    # Link2 is between the first joint and the second joint (end of the second link)
    x2, y2 = arm_positions[2]

    # Define the boxes (position, width, height, and orientation)
    link1_box = {
        'position': [(x0 + x1) / 2, (y0 + y1) / 2],  # Center of link1
        'width': link1_length,
        'height': link_width,
        'orientation': theta0  # Orientation of the first link is theta0
    }

    link2_box = {
        'position': [(x1 + x2) / 2, (y1 + y2) / 2],  # Center of link2
        'width': link2_length,
        'height': link_width,
        'orientation': theta0 + theta1  # Orientation of the second link is theta0 + theta1
    }

    return [link1_box, link2_box]


def collision_free_conf(robot_type, robot_configuration, environment, debug=False):
    if robot_type == 'freeBody':
        for i, obstacle in enumerate(environment):
            if check_collision(robot_configuration, obstacle):
                return False
        return True
    if robot_type == 'arm':
        # Check collision at configuration of arm
         # robot_configuration should be a tuple of joint angles (theta0, theta1)
        if debug: print('Robot Config:', robot_configuration)
        theta0, theta1 = robot_configuration
        
        # Get the positions of the links using forward kinematics
        arm_positions = forward_kinematics(theta0, theta1)
        if debug: print(arm_positions)
        
        # Get the link boxes (rectangular representations of the links)
        link_boxes = get_link_boxes(arm_positions, theta0, theta1)
        if debug: print(link_boxes)
        # Check if either link collides with any obstacle
        for obstacle in environment:
            for link_box in link_boxes:
                if check_collision(link_box, obstacle, arm=True):  # Check collision for each link box
                    return False
        
        return True

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
        robot_orientation = random.uniform(0, 2 * math.pi)  # Random orientation in radians
        robot = {'position': robot_position, 'width': robot_size[0], 'height': robot_size[1], 'orientation': robot_orientation}
        
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