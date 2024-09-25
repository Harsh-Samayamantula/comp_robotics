import numpy as np
import matplotlib as plt
import scipy.stats
from utils import *
from component_1 import *


def random_rotation_matrix(naive=True):
    if naive:
        # Naive solution --> Random euler angles and multiplying rotation matrices
        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0, 2*np.pi)
        gamma = np.random.uniform(0, 2*np.pi)

        # alpha, beta, gamma = random_euler_angles()


        R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])
        
        R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]])

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(gamma), -np.sin(gamma)],
                        [0, np.sin(gamma), np.cos(gamma)]])

        res = R_z @ R_y @ R_x

        print(check_SOn(res))

        return res


    else:
        # Efficient method using householder matrix
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        x3 = np.random.uniform(0, 1)

        theta = 2*np.pi*x1 # rotation about pole
        phi = 2*np.pi*x2 # direction to deflect pole
        z = x3 # pole deflection

        V = np.array([np.cos(phi)*np.sqrt(z),
                    np.sin(phi)*np.sqrt(z),
                    np.sqrt(1-z)])
        
        B = np.array([[np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
        
        
        # M = 2 * V @ V.T - np.eye(3)
        M = np.eye(3) - 2 * np.outer(V, V)

        res = M @ B

        if np.linalg.det(res) < 0:
            res[:, 0] = -res[:, 0] 


        print(check_SOn(res))

        return res


def random_quaternion(naive=True):
    if naive:
        alpha = np.random.uniform(0, 2*np.pi)  # Yaw
        beta = np.random.uniform(0, 2*np.pi)     # Pitch
        gamma = np.random.uniform(0, 2*np.pi)  # Roll

        w = np.cos(alpha/2) * np.cos(beta/2) * np.cos(gamma/2) + np.sin(alpha/2) * np.sin(beta/2) * np.sin(gamma/2)
        x = np.sin(alpha/2) * np.cos(beta/2) * np.cos(gamma/2) - np.cos(alpha/2) * np.sin(beta/2) * np.sin(gamma/2)
        y = np.cos(alpha/2) * np.sin(beta/2) * np.cos(gamma/2) + np.sin(alpha/2) * np.cos(beta/2) * np.sin(gamma/2)
        z = np.cos(alpha/2) * np.cos(beta/2) * np.sin(gamma/2) - np.sin(alpha/2) * np.sin(beta/2) * np.cos(gamma/2)
        
        q =  np.array([w, x, y, z])

        print(check_quaternion(q))

        return q
        
    else:
        s = np.random.uniform()
        sigma1 = np.sqrt(1-s)
        sigma2 = np.sqrt(s)

        theta1 = np.random.uniform(0, 2*np.pi)
        theta2 = np.random.uniform(0, 2*np.pi)

        w = np.cos(theta2)*sigma2
        x = np.sin(theta1)*sigma1
        y = np.cos(theta1)*sigma1
        z = np.sin(theta2)*sigma2

        q = np.array([w, x, y, z])

        print(check_quaternion(q))

        return q
    

def quaternion_multiply(q1, q0):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    
def visualize_rotation(M, quaternion=False):
    v0 = np.array([0,0,1]) # North Pole
    epsilon = 0.01
    v1 = np.array([0, epsilon, 0]) + v0

    if quaternion:
        q = M
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v0_q = np.concatenate(([0], v0))
        v1_q = np.concatenate(([0], v1))
        
        v0_prime = quaternion_multiply(quaternion_multiply(q, v0_q), q_conj)
        v1_prime = quaternion_multiply(quaternion_multiply(q, v1_q), q_conj)
    else:
        v0_prime = v0 @ M
        v1_prime = v1 @ M - v0
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(v0_prime[0], v0_prime[1], v0_prime[2], v1_prime[0], v1_prime[1], v1_prime[2], color='m', label='v1_prime (rotated)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    plt.show()

print(random_rotation_matrix(naive=True))
print(random_rotation_matrix(naive=False))

print(random_quaternion(naive=True))
print(random_quaternion(naive=False))