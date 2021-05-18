import math
import numpy as np


def euclidian_length(v):
    return np.linalg.norm(v)


def normalize(a):
    length = np.linalg.norm(a)
    if length < 0.000001:
        return np.array([0, 0, 0])
    return a / length


def glm_mix(v1, v2, a):
    return np.array(((1 - a) * v1 + a * v2))


def z_angle(v1):
    v = np.array([v1[0], 0.0, v1[2]])
    v = normalize(v)
    return math.atan2(v[0], v[-1])


def mix_directions(v1, v2, a):
    if v1.all(0) and v2.all(0):
        return v1
    v1 = normalize(v1)
    v2 = normalize(v2)
    omega = math.acos(max(min(np.dot(v1, v2), 1), -1))
    sinom = math.sin(omega)
    if sinom < 0.000001:
        return v1
    slerp = math.sin((1 - a) * omega) / sinom * v1 + math.sin(a * omega) / sinom * v2
    return normalize(slerp)


# Angle in radians
def rot_around_z_3d(vector, angle, inverse=False):
    mat = np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])
    if inverse:
        mat = np.linalg.inv(mat)
    return np.matmul(mat, vector)


def xz_to_x0yz(arr, axis=None):
    """
    To be used for root projections as they are expected to lie on the ground i.e zeros in y axis.
    Converts input from x, z i.e 2D to x, y, z i.e 3D.
    :param arr: np.array(2) or np.array(n_rows,2)
    :param axis: None       or 1 (when adding entire array of zeros to array of shape (n_rows,2))
    :return: np.array(3)    or np.array(n_rows, 3)
    """
    return np.insert(arr, 1, 0, axis)
