import numpy as np
import matplotlib
import scipy.stats
from utils import *

def check_SOn(m, epsilon=0.01):
    # Orthogonality: m^tm = I
    # Determinant = 1
    n = m.shape[0]
    identity = np.eye(n)
    return np.allclose(m.T @ m, identity, atol=epsilon) and np.isclose(np.linalg.det(m), 1, atol=epsilon)
    

def check_quaternion(v, epsilon=0.01):
    # Check unit quaternion --> norm must be 1
    return np.isclose(np.linalg.norm(v), 1, atol=epsilon)

def check_SEn(m, epsilon=0.01):
    # Check SO(n)
    n = m.shape[0] - 1
    rotation = m[:n, :n]
    translation = m[:n, n]
    valid_rotation = check_SOn(rotation)

    # Check last row
    valid_row = np.allclose(m[n, :], [0]*n + [1], atol=epsilon)

    return valid_rotation and valid_row

def correct_SOn(m, epsilon=0.01):
    if check_SOn(m): return m

    # Correcting Orthogonality
    U, S, Vt = np.linalg.svd(m)
    corrected_matrix = U @ Vt
    # Correcting determinant
    if np.linalg.det(corrected_matrix) < 0:
        U[:, -1] *= -1
        corrected_matrix = U @ Vt
    return corrected_matrix
    

def correct_SEn(m, epsilon=0.01):
    if check_SEn(m): return m

    n = m.shape[0] - 1
    rotation = m[:n, :n]
    corrected_rotation = correct_SOn(rotation)

    # Rebuild
    res_matrix = np.eye(n+1)
    res_matrix[:n, :n] = corrected_rotation
    res_matrix[:n, n] = m[:n, n]

    return res_matrix



def correct_quaternion(v, epsilon=0.01):
    if check_quaternion(v): return v

    norm = np.linalg.norm(v)
    if np.abs(norm - 1) > epsilon:
        return v / norm
    return v




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
        
        
        M = 2 * V @ V.T - np.eye(3)

        res = M @ B

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