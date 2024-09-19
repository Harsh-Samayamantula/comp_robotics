import numpy
import matplotlib
import scipy.stats


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


    
