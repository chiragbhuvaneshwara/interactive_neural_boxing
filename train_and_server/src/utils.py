import math
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cubic(y0, y1, y2, y3, mu):
    return (
            (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
            (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
            (-0.5 * y0 + 0.5 * y2) * mu +
            (y1))


def euclidian_length(v):
    # s = 0.0
    # for a in v:
    #     print(a*a)
    #     s += a * a
    # return math.sqrt(s)
    return np.linalg.norm(v)


# helpers:
def normalize(a):
    length = np.linalg.norm(a)
    # normal_array = an_array / norm
    # length = euclidian_length(a)
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
    # mat = np.array([
    #     [math.cos(angle), -math.sin(angle), 0],
    #     [math.sin(angle), math.cos(angle), 0],
    #     [0, 0, 1],
    # ])
    if inverse:
        mat = np.linalg.inv(mat)
    # mat = mat.transpose()
    return np.matmul(mat, vector)


def quat_to_mat(q):
    qr = q[3]
    qi = q[0]
    qj = q[1]
    qk = q[2]
    s = 1
    return np.array([
        [1 - 2 * s * (qj * qj + qk * qk), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
        [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi * qi + qk * qk), 2 * s * (qj * qk - qi * qr)],
        [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi * qi + qj * qj)]
    ])


def mat_to_quat(m):
    # print(m)
    qw = math.sqrt(1.0 + m[0][0] + m[1][1] + m[2][2]) / 2.0
    qx = (m[2][1] - m[1][2]) / (4 * qw)
    qy = (m[0][2] - m[2][0]) / (4 * qw)
    qz = (m[1][0] - m[0][1]) / (4 * qw)
    return np.array((qx, qy, qz, qw))


def global_to_local_pos(pos, root_pos, root_rot):
    return rot_around_z_3d(pos - root_pos, root_rot, inverse=True)  # self.char.joint_positions[i]#


def xz_to_x0yz(arr, axis=None):
    """
    To be used for root projections as they are expected to lie on the ground i.e zeros in y axis.
    Converts input from x, z i.e 2D to x, y, z i.e 3D.
    :param arr: np.array(2) or np.array(n_rows,2)
    :param axis: None       or 1 (when adding entire array of zeros to array of shape (n_rows,2))
    :return: np.array(3)    or np.array(n_rows, 3)
    """
    return np.insert(arr, 1, 0, axis)
