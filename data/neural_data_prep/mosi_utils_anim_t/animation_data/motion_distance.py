import numpy as np
from math import sqrt
from ..external.transformations import euler_matrix


def _align_point_clouds_2D(a, b, weights):
    '''
     Finds aligning 2d transformation of two equally sized point clouds.
     from Motion Graphs paper by Kovar et al.
     Parameters
     ---------
     *a: list
     \t point cloud
     *b: list
     \t point cloud
     *weights: list
     \t weights of correspondences
     Returns
     -------
     *theta: float
     \t angle around y axis in radians
     *offset_x: float
     \t
     *offset_z: float

     '''
    if len(a) != len(b):
        raise ValueError("two point cloud should have the same number points: "+str(len(a))+","+str(len(b)))
    n_points = len(a)
    numerator_left = 0
    denominator_left = 0
    weighted_sum_a_x = 0
    weighted_sum_b_x = 0
    weighted_sum_a_z = 0
    weighted_sum_b_z = 0
    sum_of_weights = 0.0
    #    if not weights:
    #        weight = 1.0/n_points # todo set weight base on joint level
    for index in range(n_points):
        numerator_left += weights[index] * (a[index][0] * b[index][2] -
                                            b[index][0] * a[index][2])
        denominator_left += weights[index] * (a[index][0] * b[index][0] +
                                              a[index][2] * b[index][2])
        sum_of_weights += weights[index]
        weighted_sum_a_x += weights[index] * a[index][0]
        weighted_sum_b_x += weights[index] * b[index][0]
        weighted_sum_a_z += weights[index] * a[index][2]
        weighted_sum_b_z += weights[index] * b[index][2]
    numerator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_z -
         weighted_sum_b_x * weighted_sum_a_z)
    numerator = numerator_left - numerator_right
    denominator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_x +
         weighted_sum_a_z * weighted_sum_b_z)
    denominator = denominator_left - denominator_right
    theta = np.arctan2(numerator, denominator)
    offset_x = (weighted_sum_a_x - weighted_sum_b_x *
                np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x *
                np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights

    return theta, offset_x, offset_z


def _point_cloud_distance(a, b):
    """ Calculates the distance between two point clouds with equal length and
    corresponding indices
    """
    assert len(a) == len(b)
    distance = 0
    n_points = len(a)
    for i in range(n_points):
        d = [a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]]
        distance += sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    return distance / n_points


def convert_quat_frame_to_point_cloud(skeleton, frame, joints=None):
    points = []
    if joints is None:
        joints = [k for k, n in list(skeleton.nodes.items()) if len(n.children) > 0]
    for j in joints:
        p = skeleton.nodes[j].get_global_position(frame)
        points.append(p)
    return points


def _transform_point_cloud(point_cloud, theta, offset_x, offset_z):
    """ Transforms points in a point cloud by a rotation around the y axis and a translation
        along x and z
    """
    transformed_point_cloud = []
    for p in point_cloud:
        if p is not None:
            m = euler_matrix(0,np.radians(theta), 0)
            m[0, 3] = offset_x
            m[2, 3] = offset_z
            p = p.tolist()
            new_point = np.dot(m, p + [1])[:3]
            transformed_point_cloud.append(new_point)
    return transformed_point_cloud


def _transform_invariant_point_cloud_distance(pa,pb, weights=None):
    if weights is None:
        weights = [1.0] * len(pa)
    theta, offset_x, offset_z = _align_point_clouds_2D(pa, pb, weights)
    t_b = _transform_point_cloud(pb, theta, offset_x, offset_z)
    distance = _point_cloud_distance(pa, t_b)
    return distance


def frame_distance_measure(skeleton, a, b, weights=None):
    """ Converts frames into point clouds and calculates
        the distance between the aligned point clouds.

    Parameters
    ---------
    *skeleton: Skeleton
    \t skeleton used for forward kinematics
    *a: np.ndarray
    \t the parameters of a single frame representing a skeleton pose.
    *b: np.ndarray
    \t the parameters of a single frame representing a skeleton pose.
    *weights: list of floats
    \t weights giving the importance of points during the alignment and distance

    Returns
    -------
    Distance representing similarity of the poses.
    """
    assert len(a) == len(b)
    pa = convert_quat_frame_to_point_cloud(skeleton, a)
    pb = convert_quat_frame_to_point_cloud(skeleton, b)
    return _transform_invariant_point_cloud_distance(pa,pb, weights)
