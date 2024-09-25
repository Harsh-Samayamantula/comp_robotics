import numpy as np
import matplotlib
import scipy.stats

def random_euler_angles():
    theta = 2*np.pi * np.random.uniform() - np.pi
    phi = np.arccos(1-2*np.random.uniform()) + np.pi/2

    if np.random.uniform() < 0.5:
        if phi < np.pi:
            phi += np.pi
        else:
            phi -= np.pi
    
    iota = 2 * np.pi * np.random.uniform() - np.pi

    return theta, phi, iota


