# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:11:35 2015

@author: Erik, Han, Markus
"""
import os
import numpy as np
from math import sqrt, radians, sin, cos, isnan
from copy import deepcopy
import glob
from .bvh import BVHReader, BVHWriter
#from .skeleton_builder import SkeletonBuilder # deprecated
from scipy import stats  # linear regression
from .quaternion_frame import QuaternionFrame
from collections import OrderedDict
from .body_plane import BodyPlane
from .constants import LEN_EULER, LEN_ROOT, LEN_QUAT
from .quaternion import Quaternion
from ..external.transformations import quaternion_matrix, euler_from_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_multiply, \
    quaternion_about_axis, \
    rotation_matrix, \
    quaternion_conjugate, \
    quaternion_inverse, \
    rotation_from_matrix, \
    quaternion_slerp, \
    quaternion_conjugate,  quaternion_inverse, rotation_from_matrix
# use_euler_fk = False
# try:
#     import fk3
#     use_euler_fk = True
# except:
#     print("Could not find euler FK-Extension")

from .constants import *

# if use_euler_fk:
#     FK_FUNCS = [
#         fk3.one_joint_fk,
#         fk3.two_joints_fk,
#         fk3.three_joints_fk,
#         fk3.four_joints_fk,
#         fk3.five_joints_fk,
#         fk3.six_joints_fk,
#         fk3.seven_joints_fk,
#         fk3.eight_joints_fk,
#     ]
#     FK_FUNC_JACS = [
#         fk3.one_joint_fk_jacobian,
#         fk3.two_joints_fk_jacobian,
#         fk3.three_joints_fk_jacobian,
#         fk3.four_joints_fk_jacobian,
#         fk3.five_joints_fk_jacobian,
#         fk3.six_joints_fk_jacobian,
#         fk3.seven_joints_fk_jacobian,
#         fk3.eight_joints_fk_jacobian,
#     ]


def rotation_order_to_string(rotation_order):
    r_order_string = "r"
    for c in rotation_order:
        if c == "Xrotation":
            r_order_string += "x"
        elif c == "Yrotation":
            r_order_string += "y"
        elif c == "Zrotation":
            r_order_string += "z"
    return r_order_string


def extract_root_positions_from_frames(frames):
    roots = []
    for i in range(len(frames)):
        position = np.array([frames[i][0], frames[i][1], frames[i][2]])
        roots.append(position)
    return np.array(roots)

def quaternion_inv(q):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # """
    # Inverse of quaternion q
    # Args:
    #         q: (qw, qx, qy, qz)
    # """
    # w, x, y, z = q
    # d = w*w + x*x + y*y + z*z
    # q_inv = (w/d, -x/d, -y/d, -z/d)
    # return q_inv

def get_arc_length_from_points(points):
    """
    Note: accuracy depends on the granulariy of points
    """
    points = np.asarray(points)
    arc_length = 0.0
    last_p = None
    for p in points:
        if last_p is not None:
            delta = p - last_p
            arc_length += np.linalg.norm(delta)
        last_p = p
    return arc_length


def get_smoothed_quaternion_motions_from_euler_motions(euler_motions, ref_bvh):
    '''
    convert Euler animation to Quaternion animation
    :param euler_motions (dictionary): filename: euler frames
    :param ref_skeleton (BVHReader):
    :return:
    '''
    quat_frames = OrderedDict()
    for filename, values in euler_motions.iteritems():
        quat_frames[filename] = convert_euler_frames_to_quaternion_frames(ref_bvh, values)
    ## perform smooth on quaternion frames
    smoothed_quat_frames = quaternion_data_smoothing(quat_frames)
    return smoothed_quat_frames


def quaternion_data_smoothing(quat_frames):
    smoothed_quat_frames = {}
    filenames = quat_frames.keys()
    smoothed_quat_frames_data = np.asarray(copy.deepcopy(quat_frames.values()))
    print('quaternion frame data shape: ', smoothed_quat_frames_data.shape)
    assert len(smoothed_quat_frames_data.shape) == 3, ('The shape of quaternion frames is not correct!')
    n_samples, n_frames, n_dims = smoothed_quat_frames_data.shape
    n_joints = (n_dims - 3)/4
    for i in range(n_joints):
        ref_quat = smoothed_quat_frames_data[0, 0, i*LEN_QUAT + LEN_ROOT : (i+1)*LEN_QUAT + LEN_ROOT]
        for j in range(1, n_samples):
            test_quat = smoothed_quat_frames_data[j, 0, i*LEN_QUAT + LEN_ROOT : (i+1)*LEN_QUAT + LEN_ROOT]
            if not areQuatClose(ref_quat, test_quat):
                smoothed_quat_frames_data[j, :, i*LEN_QUAT + LEN_ROOT : (i+1)*LEN_QUAT + LEN_ROOT] *= -1
    for i in range(len(filenames)):
        smoothed_quat_frames[filenames[i]] = smoothed_quat_frames_data[i]
    return smoothed_quat_frames


def areQuatClose(quat1, quat2):
    dot = np.dot(quat1, quat2)
    if dot < 0.0:
        return False
    else:
        return True


def compute_average_quaternions(quaternions):
    '''

    :param quaternions: a set of Quaternion
    :return:
    '''
    quaternions = [q.normalized() for q in quaternions]  ## normalize quaternion

    weight = 1.0 / len(quaternions)

    mean_q = Quaternion(weight * quaternions[0].toVector())

    for i in range(len(quaternions)-1):
        if not areQuatClose(mean_q.normalized().toVector(), quaternions[i+1].toVector()):
            quaternions[i+1] = quaternions[i+1].invert_rotation()
        mean_q.x += weight * quaternions[i+1].x
        mean_q.y += weight * quaternions[i+1].y
        mean_q.z += weight * quaternions[i+1].z
        mean_q.w += weight * quaternions[i+1].w

    return mean_q.normalized()


def get_step_length(frames):
    """

    :param frames: a list of euler or quaternion frames
    :return step_len: travelled distance of root joint
    """
    root_points = extract_root_positions(frames)
    step_len = get_arc_length_from_points(root_points)
    return step_len


def inverse_pose_transform(translation, euler_angles):
    inv_translation = -translation[0],-translation[1],-translation[2]
    inv_euler_angles = -euler_angles[0],-euler_angles[1],-euler_angles[2]
    return inv_translation, inv_euler_angles


def euler_to_quaternion(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER, filter_value=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]

    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in degrees
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    """
    # convert euler angle from degree into radians
    return euler_to_quaternion_rad(np.deg2rad(euler_angles), rotation_order, filter_value)


def euler_to_quaternion_rad(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER, filter_value=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]

    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in radian
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    """
    # convert euler angles into rotation matrix, then convert rotation matrix
    # into quaternion
    assert len(euler_angles) == 3, ('The length of euler angles should be 3!')
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rxyz')
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rxzy')
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='ryxz')
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='ryzx')
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rzxy')
        elif rotation_order[1] == 'Yrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rzyx')
    # convert rotation matrix R into quaternion vector (qw, qx, qy, qz)
    q = quaternion_from_matrix(R)
    # filter the quaternion http://physicsforgames.blogspot.de/2010/02/quaternions.html
    if filter_value:
        dot = np.sum(q)
        if dot < 0:
            q = -q
    return q[0], q[1], q[2], q[3]


def quaternion_to_euler_smooth(q, euler_ref, rotation_order=DEFAULT_ROTATION_ORDER):
    """

    :param q: list of floats, [qw, qx, qy, qz]
    :param euler_ref: reference euler angle in degree, pick the euler value which is closer to reference rotation
    :param rotation_order: list of str
    :return: euler angles in degree
    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # len_angle = len(euler_ref)
    # euler_angles = quaternion_to_euler(q, rotation_order)
    # for i in range(len_angle):
    #     if euler_ref[i] < 0:
    #         diff1 = abs(euler_ref[i] - euler_angles[i])
    #         diff2 = abs(euler_angles[i] - 360 - euler_ref[i])
    #         if diff2 < diff1:
    #             euler_angles[i] -= 360
    #     else:
    #         diff1 = abs(euler_ref[i] - euler_angles[i])
    #         diff2 = abs(euler_angles[i] + 360 -euler_ref[i])
    #         if diff2 < diff1:
    #             euler_angles[i] += 360
    # return euler_angles


def quaternion_to_euler_rad(q, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    Parameters
    ----------
    * q: list of floats
    \tQuaternion vector with form: [qw, qx, qy, qz]

    Return
    ------
    * euler_angles: list
    \tEuler angles in radian with specified order
    """
    q = np.asarray(q)
    # normalize quaternion vector
    q /= np.linalg.norm(q)
    # create rotation matrix from quaternion
    Rq = quaternion_matrix(q)
    # map rotation order
    r_order_string = rotation_order_to_string(rotation_order)
    return euler_from_matrix(Rq, r_order_string)


def quaternion_to_euler(q, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    Parameters
    ----------
    * q: list of floats
    \tQuaternion vector with form: [qw, qx, qy, qz]

    Return
    ------
    * euler_angles: list
    \tEuler angles in degree with specified order
    """
    return np.rad2deg(quaternion_to_euler_rad(q, rotation_order)).tolist()


def convert_euler_frames_to_quaternion_frames(bvhreader, euler_frames, filter_joints=True, animated_joints=None):
    """
    :param bvhreader: a BVHReader instance to store skeleton information
    :param euler_frames: a list of euler frames
    :return: a list of quaternion frames
    """
    quat_frames = []
    last_quat_frame = None
    for frame in euler_frames:
        quat_frame = QuaternionFrame(bvhreader, frame, ignore_bip_joints=filter_joints, animated_joints=animated_joints)
        if last_quat_frame is not None:
            for joint_name in list(bvhreader.node_names.keys()):
                if joint_name in list(quat_frame.keys()):
                    quat_frame[joint_name] = check_quat(
                        quat_frame[joint_name],
                        last_quat_frame[joint_name])
        last_quat_frame = quat_frame
        root_translation = frame[0:3]
        quat_frame_values = [root_translation, ] + list(quat_frame.values())
        quat_frame_values = convert_quat_frame_value_to_array(quat_frame_values)
        quat_frames.append(quat_frame_values)
    return quat_frames


def convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame):
    if type(euler_frame) != list:
        euler_frame = list(euler_frame)
    reduced_euler_frame = euler_frame[:LEN_ROOT]
    for node_name in bvhreader.node_names:
        if not node_name.startswith('Bip') and 'EndSite' not in node_name:
            x_idx = bvhreader.node_channels.index((node_name, 'Xrotation'))
            reduced_euler_frame += euler_frame[x_idx: x_idx + LEN_EULER]
    return reduced_euler_frame


def convert_euler_frames_to_reduced_euler_frames(bvhreader, euler_frames):
    reduced_euler_frames = []
    for euler_frame in euler_frames:
        reduced_euler_frames.append(convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame))
    return reduced_euler_frames


def convert_quat_frame_value_to_array(quat_frame_values):
    n_channels = len(quat_frame_values)
    quat_frame_value_array = []
    for item in quat_frame_values:
        if not isinstance(item, list):
            item = list(item)
        quat_frame_value_array += item
    return quat_frame_value_array


def check_quat(test_quat, ref_quat):
    """check locomotion_synthesis_test quat needs to be filpped or not
    """
    test_quat = np.asarray(test_quat)
    ref_quat = np.asarray(ref_quat)
    dot = np.dot(test_quat, ref_quat)
    if dot < 0:
        test_quat = - test_quat
    return test_quat.tolist()


def convert_quaternion_frames_to_euler_frames(quaternion_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------

    :param quaternion_frames:
     * quaternion_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------

    * euler_frames: numpy array
    \tEuler frames
    """

    def gen_4_tuples(it):
        """Generator of n-tuples from iterable"""

        return list(zip(it[0::4], it[1::4], it[2::4], it[3::4]))

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""

        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler_frame += quaternion_to_euler(quaternion)

        return euler_frame

    euler_frames = list(map(get_euler_frame, quaternion_frames))

    return np.array(euler_frames)


def convert_quaternion_to_euler(quaternion_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------

     * quaternion_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------

    * euler_frames: numpy array
    \tEuler frames
    """

    def gen_4_tuples(it):
        """Generator of n-tuples from iterable"""

        return list(zip(it[0::4], it[1::4], it[2::4], it[3::4]))

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""

        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler_frame += quaternion_to_euler(quaternion)

        return euler_frame

    euler_frames = list(map(get_euler_frame, quaternion_frames))

    return np.array(euler_frames)


def euler_substraction(theta1, theta2):
    ''' compute the angular distance from theta2 to theta1, positive value is anti-clockwise, negative is clockwise
    @param theta1, theta2: angles in degree
    '''
    theta1 %= 360
    theta2 %= 360
    if theta1 > 180:
        theta1 -= 360
    elif theta1 < -180:
        theta1 += 360

    if theta2 > 180:
        theta2 -= 360
    elif theta2 < - 180:
        theta2 += 360

    theta = theta1 - theta2
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    if theta > 180 or theta < - 180:
        raise ValueError(' exception value')
    return theta


def get_cartesian_coordinates_from_quaternion(skeleton, node_name, quaternion_frame, return_gloabl_matrix=False):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------
    * node_name: String
    \tName of node
     * skeleton: Skeleton
    \tBVH data structure read from a file

    """
    if skeleton.node_names[node_name]["level"] == 0:
        root_frame_position = quaternion_frame[:3]
        root_node_offset = skeleton.node_names[node_name]["offset"]

        return [t + o for t, o in
                zip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

        offsets = [skeleton.node_names[nodename]["offset"]
                   for nodename in chain_names]
        root_position = quaternion_frame[:3].flatten()
        offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]

        j_matrices = []
        count = 0
        for node_name in chain_names:
            index = skeleton.node_name_frame_map[node_name] * 4 + 3
            j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
            j_matrix[:, 3] = offsets[count] + [1]
            j_matrices.append(j_matrix)
            count += 1

        global_matrix = np.identity(4)
        for j_matrix in j_matrices:
            global_matrix = np.dot(global_matrix, j_matrix)
        if return_gloabl_matrix:
            return global_matrix
        else:
            point = np.array([0, 0, 0, 1])
            point = np.dot(global_matrix, point)
            return point[:3].tolist()


def get_cartesian_coordinates_for_plam_quaternion(
        skeleton,
        quat_frame,
        node_name='Left'):
    if node_name == 'Left':
        node_name = 'LeftHand'
        Finger2 = 'Bip01_L_Finger2'
        Finger21 = 'Bip01_L_Finger21'
        Finger2_offset = [9.55928, -0.145352, -0.186424]
        Finger2_angles = [-0.0, 0.0, 0.0]
        Finger21_offset = [3.801407, 0.0, 0.0]
        Finger21_angles = [-0.0, 0.0, 0.0]
    elif node_name == 'Right':
        node_name = 'RightHand'
        Finger2 = 'Bip01_R_Finger2'
        Finger21 = 'Bip01_R_Finger21'
        Finger2_offset = [9.559288, 0.145353, -0.186417]
        Finger2_angles = [-0.0, 0.0, 0.0]
        Finger21_offset = [3.801407, 0.0, 0.0]
        Finger21_angles = [-0.0, 0.0, 0.0]
    else:
        raise ValueError('Unknown node name!')
    global_matrix = get_cartesian_coordinates_from_quaternion(
        skeleton,
        node_name,
        quat_frame,
        return_gloabl_matrix=True)
    quat_finger2 = euler_to_quaternion(Finger2_angles)
    transmat_finger2 = quaternion_matrix(quat_finger2)
    transmat_finger2[:3, 3] = Finger2_offset[:]
    global_matrix = np.dot(global_matrix, transmat_finger2)
    quat_finger21 = euler_to_quaternion(Finger21_angles)
    transmat_finger21 = quaternion_matrix(quat_finger21)
    transmat_finger21[:3, 3] = Finger21_offset[:]
    global_matrix = np.dot(global_matrix, transmat_finger21)
    return global_matrix[:3, 3]

# if use_euler_fk:
#     def get_cartesian_coordinates_from_euler_full_skeleton(bvh_reader,
#                                                            skeleton,
#                                                            node_name,
#                                                            euler_frame):
#         """Return cartesian coordinates for one node at one frame, include the
#            skipped joints starting with "Bip"
#         """
#         print("Deprecation Warning: Function marked as Deprecated!")
#         pass
#         # if bvh_reader.node_names[node_name]["level"] == 0:
#         #     root_frame_position = euler_frame[:3]
#         #     root_node_offset = bvh_reader.node_names[node_name]["offset"]

#         #     return [t + o for t, o in
#         #             zip(root_frame_position, root_node_offset)]

#         # else:
#         #     # Names are generated bottom to up --> reverse
#         #     chain_names = list(skeleton.gen_all_parents(node_name))
#         #     chain_names.reverse()
#         #     chain_names += [node_name]  # Node is not in its parent list

#         #     eul_angles = []
#         #     index = 0
#         #     for nodename in chain_names:
#         #         indeces = []
#         #         for channel in bvh_reader.node_names[nodename]["channels"]:
#         #             if channel.endswith("rotation"):
#         #                 indeces.append(
#         #                     bvh_reader.node_channels.index((nodename, channel)))
#         #         eul_angles.append(euler_frame[indeces])
#         #         index += 1

#         #     # bvh_reader.node_names.keys().index("RightShoulder")*3 +
#         #     # 3,len(euler_frame)
#         #     rad_angles = (list(map(radians, eul_angle)) for eul_angle in eul_angles)

#         #     thx, thy, thz = list(map(list, list(zip(*rad_angles))))

#         #     offsets = [bvh_reader.node_names[nodename]["offset"]
#         #                for nodename in chain_names]

#         #     # Add root offset to frame offset list
#         #     root_position = euler_frame[:3]
#         #     offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]

#         #     ax, ay, az = list(map(list, zip(*offsets)))

#         # # f_idx identifies the kinematic forward transform function
#         # # This does not lead to a negative index because the root is
#         # # handled separately

#         # f_idx = len(ax) - 2
#         # if len(ax) - 2 < len(FK_FUNCS):
#         #     return FK_FUNCS[f_idx](ax, ay, az, thx, thy, thz)
#         # else:
#         #     return [0, 0, 0]

def get_cartesian_coordinates_for_plam_euler(
        skeleton,
        euler_frame,
        node_name='Left'):
    if node_name == 'Left':
        node_name = 'LeftHand'
        Finger2 = 'Bip01_L_Finger2'
        Finger21 = 'Bip01_L_Finger21'
        Finger2_offset = [9.55928, -0.145352, -0.186424]
        Finger2_angles = [-0.0, 0.0, 0.0]
        Finger21_offset = [3.801407, 0.0, 0.0]
        Figner21_angles = [-0.0, 0.0, 0.0]
    elif node_name == 'Right':
        node_name = 'RightHand'
        Finger2 = 'Bip01_R_Finger2'
        Finger21 = 'Bip01_R_Finger21'
        Finger2_offset = [9.559288, 0.145353, -0.186417]
        Finger2_angles = [-0.0, 0.0, 0.0]
        Finger21_offset = [3.801407, 0.0, 0.0]
        Figner21_angles = [-0.0, 0.0, 0.0]
    else:
        raise ValueError('Unknown node name!')
    chain_names = list(skeleton.gen_all_parents(node_name))
    chain_names.reverse()
    chain_names += [node_name]  # Node is not in its parent list
    global_trans = np.eye(4)
    global_trans[:3, 3] = euler_frame[:3]
    eul_angles = []
    for nodename in chain_names:
        index = skeleton.node_name_map[nodename] * 3 + 3
        eul_angles.append(euler_frame[index:index + 3])
    eul_angles.append(Finger2_angles)
    eul_angles.append(Figner21_angles)
    offsets = [skeleton.node_names[nodename]["offset"]
               for nodename in chain_names]
    offsets.append(Finger2_offset)
    offsets.append(Finger21_offset)
    chain_names.append(Finger2)
    chain_names.append(Finger21)
    for i in range(len(chain_names)):
        rot_angles = eul_angles[i]
        translation = offsets[i]
        rot_angles_rad = np.deg2rad(rot_angles)
        rotmat = euler_matrix(rot_angles_rad[0],
                              rot_angles_rad[1],
                              rot_angles_rad[2],
                              'rxyz')
        transmat = np.eye(4)
        transmat[:3, 3] = translation[:]
        global_trans = np.dot(global_trans, transmat)
        global_trans = np.dot(global_trans, rotmat)
    return global_trans[:3, 3]

# if use_euler_fk:
#     def get_cartesian_coordinates_from_euler(skeleton, node_name, euler_frame):
#         """Returns cartesian coordinates for one node at one frame. Modified to
#          handle frames with omitted values for joints starting with "Bip"

#         Parameters
#         ----------

#         * node_name: String
#         \tName of node
#          * skeleton: Skeleton
#         \t skeleton structure read from a file
#         * frame_number: Integer
#         \tAnimation frame number that gets extracted

#         """
#         print("Deprecation Warning: Function marked as Deprecated!")
#         pass
#         # if skeleton.node_names[node_name]["level"] == 0:
#         #     root_frame_position = euler_frame[:3]
#         #     root_node_offset = skeleton.node_names[node_name]["offset"]

#         #     return [t + o for t, o in
#         #             zip(root_frame_position, root_node_offset)]

#         # else:
#         #     # Names are generated bottom to up --> reverse
#         #     chain_names = list(skeleton.gen_all_parents(node_name))
#         #     chain_names.reverse()
#         #     chain_names += [node_name]  # Node is not in its parent list
#         #     eul_angles = []
#         #     for nodename in chain_names:
#         #         index = skeleton.node_name_frame_map[nodename] * 3 + 3
#         #         eul_angles.append(euler_frame[index:index + 3])
#         #     rad_angles = (list(map(radians, eul_angle)) for eul_angle in eul_angles)

#         #     thx, thy, thz = list(map(list, list(zip(*rad_angles))))

#         #     offsets = [skeleton.node_names[nodename]["offset"]
#         #                for nodename in chain_names]

#         #     # Add root offset to frame offset list
#         #     root_position = euler_frame[:3]
#         #     offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]

#         #     ax, ay, az = list(map(list, zip(*offsets)))

#         #     # f_idx identifies the kinematic forward transform function
#         #     # This does not lead to a negative index because the root is
#         #     # handled separately

#         #     f_idx = len(ax) - 2
#         #     if len(ax) - 2 < len(FK_FUNCS):
#         #         return FK_FUNCS[f_idx](ax, ay, az, thx, thy, thz)
#         #     else:
#         #         return [0, 0, 0]


def convert_euler_frame_to_cartesian_frame(skeleton, euler_frame, animated_joints=None):
    """
    converts euler frames to cartesian frames by calling get_cartesian_coordinates for each joint
    """
    if animated_joints is None:
        n_joints = len(skeleton.animated_joints)
        cartesian_frame = np.zeros([n_joints, LEN_ROOT_POS])
        for i in range(n_joints):
            cartesian_frame[i] = skeleton.nodes[skeleton.animated_joints[i]].get_global_position_from_euler(euler_frame)
    else:
        n_joints = len(animated_joints)
        cartesian_frame = np.zeros([n_joints, LEN_ROOT_POS])
        for i in range(n_joints):
            cartesian_frame[i] = skeleton.nodes[animated_joints[i]].get_global_position_from_euler(euler_frame)
    return cartesian_frame


def get_global_trans_from_euler_frames(skeleton, euler_frames, animated_joints=None):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # if animated_joints is None:
    #     animated_joints = skeleton.animated_joints
    # n_joints = len(animated_joints)
    # n_frames = len(euler_frames)
    # global_trans = np.zeros([n_frames, n_joints, 4, 4])
    # for i in range(n_frames):
    #     for j in range(n_joints):
    #         global_trans[i, j] = skeleton.nodes[animated_joints[j]].get_global_matrix_from_euler_frame(euler_frames[i])
    # return global_trans


def convert_quat_frame_to_cartesian_frame(skeleton, quat_frame, animated_joints=None):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # if animated_joints is None:
    #     n_joints = len(skeleton.animated_joints)
    #     cartesian_frame = np.zeros([n_joints, LEN_ROOT_POS])
    #     for i in range(n_joints):
    #         cartesian_frame[i] = skeleton.nodes[skeleton.animated_joints[i]].get_global_position(quat_frame)
    # else:
    #     n_joints = len(animated_joints)
    #     cartesian_frame = np.zeros([n_joints, LEN_ROOT_POS])
    #     for i in range(n_joints):
    #         cartesian_frame[i] = skeleton.nodes[animated_joints[i]].get_global_position(quat_frame)
    # return cartesian_frame


def convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frame):
    """
    Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
    """
    cartesian_frame = []
    for node_name in skeleton.node_names:
        # ignore Bip joints and end sites
        if not node_name.startswith(
                "Bip") and "children" in list(skeleton.node_names[node_name].keys()):
            cartesian_frame.append(
                get_cartesian_coordinates_from_quaternion(
                    skeleton,
                    node_name,
                    quat_frame))  # get_cartesian_coordinates2

    return cartesian_frame


def convert_quaternion_frames_to_cartesian_frames(skeleton, quat_frames):
    cartesian_frames = []
    for i in range(len(quat_frames)):
        cartesian_frames.append(convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frames[i]))
    return cartesian_frames


def transform_invariant_point_cloud_distance_2d(a, b, weights=None):
    '''

    :param a:
    :param b:
    :param weights:
    :return:
    '''
    theta, offset_x, offset_z = align_point_clouds_2D(a, b, weights)
    transformed_b = transform_point_cloud(b, theta, offset_x, offset_z)
    dist = calculate_point_cloud_distance(a, transformed_b)
    return dist


def align_point_clouds_2D(a, b, weights=None):
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
    if weights is None:
        weights = np.ones(n_points)
    numerator_left = 0.0
    denominator_left = 0.0
    #    if not weights:
    #        weight = 1.0/n_points # to do set weight base on joint level
    weighted_sum_a_x = 0.0
    weighted_sum_b_x = 0.0
    weighted_sum_a_z = 0.0
    weighted_sum_b_z = 0.0
    sum_of_weights = 0.0
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
    offset_x = (weighted_sum_a_x - weighted_sum_b_x * np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x * np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights
    return theta, offset_x, offset_z


def convert_euler_frames_to_cartesian_frames(skeleton, euler_frames, animated_joints=None):
    """
    converts to euler frames to cartesian frames
    """
    cartesian_frames = []
    for euler_frame in euler_frames:
        cartesian_frames.append(
            convert_euler_frame_to_cartesian_frame(skeleton, euler_frame, animated_joints))
    return np.array(cartesian_frames)


def convert_quat_frames_to_cartesian_frames(skeleton, quat_frames, animated_joints=None):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # cartesian_frames = []
    # for quat_frame in quat_frames:
    #     cartesian_frames.append(
    #         convert_quat_frame_to_cartesian_frame(skeleton, quat_frame, animated_joints))
    # return np.array(cartesian_frames)


def convert_quat_coeffs_to_cartesian_coeffs(elementary_action,
                                            motion_primitive,
                                            data_repo,
                                            skeleton_json,
                                            quat_coeffs_mat,
                                            knots):
    """
    convert a set of quaternion splines to cartesian splines
    :param quat_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    :return cartesian_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # from morphablegraphs.utilities import get_semantic_motion_primitive_path
    # from morphablegraphs.construction.preprocessing.quat_spline_constructor import QuatSplineConstructor
    # semantic_motion_primitive_file = get_semantic_motion_primitive_path(elementary_action,
    #                                                                     motion_primitive,
    #                                                                     data_repo)
    # quat_spline_constructor = QuatSplineConstructor(semantic_motion_primitive_file,
    #                                                 skeleton_json)
    # quat_coeffs_mat = np.asarray(quat_coeffs_mat)
    # n_samples, n_basis, n_dims = quat_coeffs_mat.shape
    # cartesian_coeffs_mat = []
    # for i in range(n_samples):
    #     quat_spline = quat_spline_constructor.create_quat_spline_from_functional_data(quat_coeffs_mat[i],
    #                                                                                   knots)
    #     cartesian_spline = quat_spline.to_cartesian()
    #     cartesian_coeffs_mat.append(cartesian_spline.coeffs)
    # return np.asarray(cartesian_coeffs_mat)


def find_aligning_transformation(skeleton, euler_frames_a, euler_frames_b):
    """
    performs alignment of the point clouds based on the poses at the end of
    euler_frames_a and the start of euler_frames_b
    Returns the rotation around y axis in radians, x offset and z offset
    """
    point_cloud_a = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_a[-1])
    point_cloud_b = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_b[0])
    weights = skeleton.get_joint_weights()
    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, weights)
    return theta, offset_x, offset_z


def rotate_around_y_axis(point, theta):
    """
    source https://www.siggraph.org/education/materials/HyperGraph/modeling/mod_tran/2drota.htm
    Parameters
    ---------
    *point: list
    \t coordinates
    *theta: float
    \t angle in radians
    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # transfomed_point = point[:]
    # transfomed_point[0] = point[0] * cos(theta) - point[2] * sin(theta)
    # transfomed_point[2] = point[2] * cos(theta) + point[0] * sin(theta)
    # return transfomed_point


def transform_point_by_quaternion(point, quaternion, offset, origin=None):
    """
    http://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    homogenous_point = np.append([0], point)
    tmp_q = quaternion_multiply(quaternion, homogenous_point)
    temp_q = quaternion_multiply(tmp_q, quaternion_conjugate(quaternion))
    new_point = [temp_q[i+1] + offset[i] + origin[i] for i in range(3)]
    return new_point


def transform_point_by_quaternion_faster(point, quaternion, offset, origin=None):
    """
    http://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    t = 2 * np.cross(quaternion[1:], point)
    new_point = np.array(point) + quaternion[0] * t + np.cross(quaternion[1:], t)
    new_point = [new_point[i] + offset[i] + origin[i] for i in range(3)]
    return new_point


def transform_point_by_quaternion_faster2(point, quaternion, offset, origin=None):
    """
    http://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    u = quaternion[1:]
    s = quaternion[0]
    new_point = 2.0 * np.dot(u, point) * u + (s*s - np.dot(u, u)) * point + 2.0 * s * np.cross(u, point)
    new_point = [new_point[i] + offset[i] + origin[i] for i in range(3)]
    return new_point


def euler_angles_to_rotation_matrix(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER):
        # generate rotation matrix based on rotation order
    assert len(euler_angles) == 3, ('The length of rotation angles should be 3')
    if round(euler_angles[0], 3) == 0 and round(euler_angles[2], 3) == 0:
        # rotation about y axis
        R = rotation_matrix(np.deg2rad(euler_angles[1]), [0, 1, 0])
    else:
        euler_angles = np.deg2rad(euler_angles)
        R = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes=rotation_order_to_string(rotation_order))
    return R


def create_transformation_matrix(translation, euler_angles, rotation_order=DEFAULT_ROTATION_ORDER):
    #m = np.eye(4,4)
    m = euler_angles_to_rotation_matrix(euler_angles, rotation_order)
    m[:3,3] = translation
    return m


def transform_point(point, euler_angles, offset, origin=None, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    rotate point around y axis and translate it by an offset
    Parameters
    ---------
    *point: numpy.array
    \t coordinates
    *angles: list of floats
    \tRotation angles in degrees
    *offset: list of floats
    \tTranslation
    """
    assert len(point) == 3, ('the point should be a list of length 3')
    # translate point to original point
    point = np.asarray(point)
    if origin is not None:
        origin = np.asarray(origin)
        point = point - origin
    point = np.insert(point, len(point), 1)
    R = euler_angles_to_rotation_matrix(euler_angles, rotation_order=rotation_order)
    rotated_point = np.dot(R, point)
    if origin is not None:
        rotated_point[:3] += origin
    #print rotated_point,
    transformed_point = rotated_point[:3] + offset
    return transformed_point


def transform_euler_frame(euler_frame, angles, offset, rotation_order=None, global_rotation=True):
    """
    Calls transform_point for the root parameters and adds theta to the y rotation
    channel of the frame.

    The offset of root is transformed by transform_point
    The orientation of root is rotated by Rotation matrix

    Parameters
    ---------
    *euler_frame: np.ndarray
    \t the parameters of a single frame
    *angles: list of floats
    \tRotation angles in degrees
    *offset: np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = ["Xrotation", "Yrotation", "Zrotation"]
    transformed_frame = deepcopy(euler_frame)
    if global_rotation:
        transformed_frame[:3] = transform_point(euler_frame[:3], angles, offset, rotation_order=rotation_order)
    else:
        transformed_frame[:3] = transform_point(euler_frame[:3], np.zeros(3), offset, rotation_order=rotation_order)
    R = euler_matrix(np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2]), axes='rxyz')
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rxyz')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rxyz'))
        elif rotation_order[1] == 'Zrotation':
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rxzy')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rxzy'))
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='ryxz')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'ryxz'))
        elif rotation_order[1] == 'Zrotation':
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='ryzx')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'ryzx'))
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rzxy')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rzxy'))
        elif rotation_order[1] == 'Yrotation':
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rzyx')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rzyx'))
    transformed_frame[3:6] = eul_angles
    return transformed_frame


def transform_euler_frames(euler_frames, angles, offset, rotation_order=None):
    """ Applies a transformation on the root joint of a list euler frames.
    Parameters
    ----------
    *euler_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    transformed_euler_frames = []
    for frame in euler_frames:
        transformed_euler_frames.append(
            transform_euler_frame(frame, angles, offset, rotation_order))
    return np.array(transformed_euler_frames)

'''
def transform_quaternion_frame(quat_frame,
                               angles,
                               offset,
                               origin=None,
                               rotation_order=None):
    """
    Calls transform_point for the root parameters and adds theta to the y
    rotation channel of the frame.
    Parameters
    ---------
    *quat_frame: np.ndarray
    \t the parameters of a single frame
    *angles: list of floats
    \tRotation angles in degrees
    *offset: np.ndarray
    \tTranslation
    """

    if rotation_order is None:
        rotation_order = ["Xrotation", "Yrotation", "Zrotation"]
    transformed_frame = quat_frame[:]
    #    original_point_copy = deepcopy(original_point)
    #    transformed_frame[:3] = transform_point(quat_frame[:3], [0, 0, 0], offset)
    # q = euler_to_quaternion(angles, rotation_order) # replace
    if round(angles[0], 3) == 0 and round(angles[2], 3) == 0:
        q = quaternion_about_axis(np.deg2rad(angles[1]), [0, 1, 0])
    else:
        q = euler_to_quaternion(angles, rotation_order)

    transformed_frame[:3] = transform_point(quat_frame[:3], angles, offset, origin)
    #transformed_frame[:3] = transform_point_by_quaternion_faster2(quat_frame[:3], q, offset, origin)
    oq = quat_frame[3:7]
    rotated_q = quaternion_multiply(q, oq)
    transformed_frame[3:7] = rotated_q
    return transformed_frame
'''


def copy_joint_rotation(src_file, target_file, target_joints, save_folder):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # src_bvhreader = BVHReader(src_file)
    # target_bvhreader = BVHReader(target_file)
    # src_skeleton = SkeletonBuilder().load_from_bvh(src_bvhreader)
    # filename = os.path.split(target_file)[-1]
    # target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    # for i in range(len(target_bvhreader.frames)):
    #     for joint in target_joints:
    #         target_joint_index = target_skeleton.nodes[joint].euler_frame_index
    #         src_joint_index = src_skeleton.nodes[joint].euler_frame_index
    #         target_bvhreader.frames[i][3+3*target_joint_index: 3+3*(target_joint_index+1)] = src_bvhreader.frames[0][3+3*src_joint_index: 3+3*(src_joint_index+1)]
    # BVHWriter(os.path.join(save_folder, filename), target_skeleton, target_bvhreader.frames,
    #           target_skeleton.frame_time, is_quaternion=False)


def get_children(joint_node, children_list=[]):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # if joint_node.children != []:
    #     for child in joint_node.children:
    #         children_list.append(child.node_name)
    #         get_children(child, children_list)


def transform_quaternion_frames(quat_frames, angles, offset, rotation_order=None):
    """ Applies a transformation on the root joint of a list quaternion frames.
    Parameters
    ----------
    *quat_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = DEFAULT_ROTATION_ORDER
    offset = np.array(offset)
    if round(angles[0], 3) == 0 and round(angles[2], 3) == 0:
        rotation_q = quaternion_about_axis(np.deg2rad(angles[1]), [0, 1, 0])
    else:
        rotation_q = euler_to_quaternion(angles, rotation_order)
    rotation_matrix = euler_angles_to_rotation_matrix(angles, rotation_order)[:3, :3]
    for frame in quat_frames:
        ot = frame[:3]
        oq = frame[3:7]
        frame[:3] = np.dot(rotation_matrix, ot) + offset
        frame[3:7] = quaternion_multiply(rotation_q, oq)
    return quat_frames


def smooth_quaternion_frames(quaternion_frames, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    quaternion_frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(quaternion_frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(quaternion_frames)
    for i in range(n_joints):
        for j in range(n_frames - 1):
            q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
            q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
            if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in range(n_frames):
        value = 0.0
        if d - w <= f < d:
            tmp = (f - d + w) / w
            value = 0.5 * tmp ** 2
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    smoothing_factors = np.array(smoothing_factors)
    new_quaternion_frames = []
    for i in range(len(quaternion_frames[0])):
        current_value = quaternion_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames


def smooth_quaternion_frames_partially(quaternion_frames, joint_parameter_indices, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    quaternion_frames: list
    \tA list of quaternion frames
    parameters_indices: list
    \tThe list of joint parameter indices that should be smoothed
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(quaternion_frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(quaternion_frames)
    for i in range(n_joints):
         for j in range(n_frames - 1):
             q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
             q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
             if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    transition_frame = float(discontinuity)
    window_size = float(window)
    smoothing_factors = []
    for frame_number in range(n_frames):
        value = 0.0
        if transition_frame - window_size <= frame_number < transition_frame:
            tmp = (frame_number - transition_frame + window_size) / window_size
            value = 0.5 * tmp ** 2
        elif transition_frame <= frame_number <= transition_frame + window_size:
            tmp = (frame_number - transition_frame + window_size) / window_size
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    #smoothing_factors = np.array(smoothing_factors)
    transition_frame = int(transition_frame)
    #new_quaternion_frames = []
    magnitude_vector = np.array(quaternion_frames[transition_frame]) - quaternion_frames[transition_frame-1]
    #print "magnitude ", magnitude_vector
    for i in joint_parameter_indices:
        joint_parameter_index = i*4+3
        for frame_index in range(n_frames):
            quaternion_frames[frame_index][joint_parameter_index] = quaternion_frames[frame_index][joint_parameter_index]+magnitude_vector[joint_parameter_index] * smoothing_factors[frame_index]
    #
    # for i in xrange(len(quaternion_frames[0])):
    #     joint_parameter_value = quaternion_frames[:, i]
    #     if i in joint_parameter_indices:
    #
    #         magnitude = joint_parameter_value[transition_frame] - joint_parameter_value[transition_frame - 1]
    #         print "smooth", i, magnitude, smoothing_factors
    #         new_value = joint_parameter_value + (magnitude * smoothing_factors)
    #         new_quaternion_frames.append(new_value)
    #     else:
    #         new_quaternion_frames.append(joint_parameter_value)
    # new_quaternion_frames = np.array(new_quaternion_frames).T
    #return quaternion_frames


def smooth_motion(euler_frames, discontinuity, window=20):
    """ Smooth a function around the given discontinuity frame

    Parameters
    ----------
    motion : AnimationController
        The motion to be smoothed
        ATTENTION: This function changes the values of the motion
    discontinuity : int
        The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
        The smoothing window
    Returns
    -------
    None.
    """
    d = float(discontinuity)
    s = float(window)

    smoothing_faktors = []
    for f in range(len(euler_frames)):
        value = 0.0
        if d - s <= f < d:
            tmp = ((f - d + s) / s)
            value = 0.5 * tmp ** 2
        elif d <= f <= d + s:
            tmp = ((f - d + s) / s)
            value = -0.5 * tmp ** 2 + 2 * tmp - 2

        smoothing_faktors.append(value)

    smoothing_faktors = np.array(smoothing_faktors)
    new_euler_frames = []
    for i in range(len(euler_frames[0])):
        current_value = euler_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        if magnitude > 180:
            magnitude -= 360
        elif magnitude < -180:
            magnitude += 360
        new_value = current_value + (magnitude * smoothing_faktors)
        new_euler_frames.append(new_value)
    new_euler_frames = np.array(new_euler_frames).T
    return new_euler_frames


def smoothly_concatenate(euler_frames_a, euler_frames_b, window_size=20):
    euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    euler_frames = smooth_motion(euler_frames, len(euler_frames_a), window_size)
    return euler_frames


def shift_euler_frames_to_ground(euler_frames, ground_contact_joints, skeleton, align_index=0):
    """
    shift all euler frames of motion to ground, which means the y-axis for
    gound contact joint should be 0
    Step 1: apply forward kinematic to compute global position for ground
            contact joint for each frame
    Setp 2: find the offset from ground contact joint to ground, and shift
            corresponding frame based on offset
    """
    # tmp_frames = deepcopy(euler_frames)
    # for frame in tmp_frames:
    #     contact_point_position = get_cartesian_coordinates_from_euler(skeleton, ground_contact_joint, frame)
    #     offset_y = contact_point_position[1]
    #     # shift root position by offset_y
    #     frame[1] = frame[1] - offset_y
    # return tmp_frames
    foot_contact_heights = []
    for joint in ground_contact_joints:
        foot_contact_heights.append(skeleton.nodes[joint].get_global_position_from_euler(euler_frames[align_index])[1])
    return transform_euler_frames(euler_frames,
                                  [0.0, 0.0, 0.0],
                                  np.array([0, -np.average(foot_contact_heights), 0]))


def shift_quat_frames_to_ground(quat_frames, ground_contact_joints, skeleton, align_index=0):
    foot_contact_heights = []
    for joint in ground_contact_joints:
        foot_contact_heights.append(skeleton.nodes[joint].get_global_position(quat_frames[align_index])[1])
    return transform_quaternion_frames(quat_frames,
                                       [0.0, 0.0, 0.0],
                                       np.array([0, -np.average(foot_contact_heights), 0]))


def lock_contact_joint(quat_frames, skeleton, contact_joint):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass

    # import copy
    # # ref_point = get_cartesian_coordinates_from_euler(skeleton, contact_joint, euler_frames[0])
    # ref_point = skeleton.nodes[contact_joint].get_global_position(quat_frames[0])
    # # ref_point[1] = 0.0
    # new_frames = copy.deepcopy(quat_frames)
    # for i in range(1, len(quat_frames)):
    #     # contact_pos = get_cartesian_coordinates_from_euler_full_skeleton(bvhreader, skeleton, contact_joint, euler_frames[i])
    #     contact_pos = skeleton.nodes[contact_joint].get_global_position(quat_frames[i])
    #     offset = np.asarray(ref_point) - np.asarray(contact_pos)
    #     new_frames[i][:3] += offset
    # return new_frames


def get_rest_pose(bvhreader, save_path):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # from morphablegraphs.animation_data import Skeleton
    # rest_pose = np.zeros(len(bvhreader.frames[0]))
    # skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # BVHWriter(save_path, skeleton, [rest_pose], bvhreader.frame_time,
    #           is_quaternion=False)


def align_frames(skeleton, euler_frames_a, euler_frames_b, smooth=True):
    """
    calls find_aligning_transformation and concatenates the frames based on the
    resulting transformation
     Parameters
    ----------
    *skeleton: Skeleton
    \tUsed to extract hierarchy information.
    *euler_frames_a: np.ndarray
    \List of frames where the rotation is represented as euler angles in degrees.
    *euler_frames_b: np.ndarray
    \List of frames where the rotation is represented as euler angles in degrees.
    *smooth: bool
    \t Sets whether or not smoothing is supposed to be applied on the at the transition.
     Returns
    -------
    *aligned_frames : np.ndarray
    \tAligned and optionally smoothed motion
    """
    theta, offset_x, offset_z = find_aligning_transformation(skeleton, euler_frames_a, euler_frames_b)

    # apply 2d transformation
    offset = np.array([offset_x, 0, offset_z])
    angles = [0, np.degrees(theta), 0]
    euler_frames_b = transform_euler_frames(euler_frames_b, angles,
                                            offset)

    # concatenate frames and optionally apply smoothing
    if smooth:
        euler_frames = smoothly_concatenate(euler_frames_a, euler_frames_b)
    else:
        euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    return euler_frames


def get_2d_pose_transform(quaternion_frames, frame_number, param_range=(3,7)):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # dir_vec = pose_orientation_quat(quaternion_frames[frame_number], param_range)
    # angle = get_rotation_angle(dir_vec, [0, -1])
    # offset_x = quaternion_frames[frame_number][0]
    # offset_z = quaternion_frames[frame_number][2]
    # return [offset_x, 0.0, offset_z], [0,angle,0]


def fast_quat_frames_transformation(quaternion_frames_a,
                                    quaternion_frames_b, param_range=(3,7)):
    #param_range = (7,11)
    dir_vec_a = pose_orientation_quat(quaternion_frames_a[-1],param_range)
    dir_vec_b = pose_orientation_quat(quaternion_frames_b[0],param_range)
    #print "dir_vector a", dir_vec_a

    #print "dir_vector b", dir_vec_b

    #find_aligning_transformation
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = quaternion_frames_a[-1][0] - quaternion_frames_b[0][0]
    offset_z = quaternion_frames_a[-1][2] - quaternion_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset


def get_quat_frame_transformation(quaternion_frame_a,
                                  quaternion_frame_b,
                                  skeleton,
                                  param_range=(3, 7),
                                  feet_contact_joints=['ball_l', 'ball_r']):
    '''
    calculate the transformation from quaternion_frame_a to quaternion_frame_b
    :param quaternion_frame_a:
    :param quaternion_frame_b:
    :param param_range:
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # dir_vec_a = pose_orientation_quat(quaternion_frame_a, param_range)
    # dir_vec_b = pose_orientation_quat(quaternion_frame_b, param_range)
    # angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    # offset_x = quaternion_frame_a[0] - quaternion_frame_b[0]
    # offset_z = quaternion_frame_a[2] - quaternion_frame_b[2]
    # lh = skeleton.nodes[feet_contact_joints[0]].get_global_position(quaternion_frame_b)[1]
    # rh = skeleton.nodes[feet_contact_joints[1]].get_global_position(quaternion_frame_b)[1]
    # offset = [offset_x, -(lh + rh)/2.0, offset_z]
    # return angle, offset


def fast_euler_frames_transformation(euler_frames_a,
                                     euler_frames_b):
    dir_vec_a = pose_orientation_euler(euler_frames_a[-1])
    dir_vec_b = pose_orientation_euler(euler_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = euler_frames_a[-1][0] - euler_frames_b[0][0]
    offset_z = euler_frames_a[-1][2] - euler_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset


def get_rotation_angle(point1, point2):
    """
    estimate the rotation angle from point2 to point1
    point1, point2 are normalized points
    rotate point2 to be the same as point1

    Parameters
    ----------
    *point1, point2: list or numpy array
    \tUnit 2d points

    Return
    ------
    *rotation_angle: float (in degree)
    \tRotation angle from point2 to point1
    """
    theta1 = point_to_euler_angle(point1)
    theta2 = point_to_euler_angle(point2)
    rotation_angle = euler_substraction(theta2, theta1)
    return rotation_angle


def point_to_euler_angle(vec):
    '''
    @brief: covert a 2D point vec = (cos, sin) to euler angle (in degree)
    The output range is [-180, 180]
    '''
    vec = np.array(vec)
    theta = np.rad2deg(np.arctan2(vec[1], vec[0]))
    return theta


def fast_quat_frames_alignment(quaternion_frames_a,
                               quaternion_frames_b,
                               smooth=True,
                               smoothing_window=DEFAULT_SMOOTHING_WINDOW_SIZE,
                               param_range=(3, 7)):
    """implement a fast frame alignment based on orientation and offset of root
       of last frame of first motion and first frame of second motion
    """

    angle, offset = fast_quat_frames_transformation(quaternion_frames_a, quaternion_frames_b, param_range)
    transformed_frames = transform_quaternion_frames(quaternion_frames_b,
                                                     [0, angle, 0],
                                                     offset)
    # concatenate the quaternion_frames_a and transformed_framess
    if smooth:
        quaternion_frames = smoothly_concatenate_quaternion_frames(
            quaternion_frames_a,
            transformed_frames,
            window_size=smoothing_window)
    else:
        quaternion_frames = np.concatenate((quaternion_frames_a,
                                            transformed_frames))
    return quaternion_frames


def fast_euler_frames_alignment(euler_frames_a,
                                euler_frames_b,
                                smooth=True,
                                smoothing_window=DEFAULT_SMOOTHING_WINDOW_SIZE):
    angle, offset = fast_euler_frames_transformation(
        euler_frames_a, euler_frames_b)
    transformed_frames = transform_euler_frames(euler_frames_b,
                                                [0, angle, 0],
                                                offset)
    # concatenate the quaternion_frames_a and transformed_framess
    if smooth:
        euler_frames = smoothly_concatenate(
            euler_frames_a,
            transformed_frames,
            window_size=smoothing_window)
    else:
        euler_frames = np.concatenate((euler_frames_a,
                                            transformed_frames))
    return euler_frames


def calculate_point_cloud_distance(a, b):
    """
    calculates the distance between two point clouds with equal length and
    corresponding distances
    """
    assert len(a) == len(b)
    distance = 0
    n_points = len(a)
    for i in range(n_points):
        d = [a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]]
        distance += sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    return distance / n_points


def rotate_and_translate_point(p, theta, offset_x, offset_z):
    """rotate and translate a 3d point, first rotation them translation
       theta is in radians
    """
    rotation_angles = [0, theta, 0]
    rotation_angles = np.rad2deg(rotation_angles)
    offset = [offset_x, 0, offset_z]
    transformed_point = transform_point(p, rotation_angles, offset)
    return transformed_point


def transform_point_cloud(point_cloud, theta, offset_x, offset_z):
    """
    transforms points in a point cloud by a rotation around y and a translation
    along x and z
    """
    transformed_point_cloud = []
    for p in point_cloud:
        if p is not None:
            transformed_point_cloud.append(
                rotate_and_translate_point(p, theta, offset_x, offset_z))
    return transformed_point_cloud


def calculate_pose_distance(skeleton, euler_frames_a, euler_frames_b):
    ''' Converts euler frames to point clouds and finds the aligning transformation
        and calculates the distance after the aligning transformation
    '''

    #    theta, offset_x, offset_z = find_aligning_transformation(bvh_reader, euler_frames_a, euler_frames_b, node_name_map)
    point_cloud_a = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_a[-1])
    point_cloud_b = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_b[0])

    weights = skeleton.joint_weights
    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, weights)
    t_point_cloud_b = transform_point_cloud(
        point_cloud_b, theta, offset_x, offset_z)
    error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    return error


def calculate_frame_distance(skeleton,
                             euler_frame_a,
                             euler_frame_b,
                             weights=None,
                             return_transform=False):
    point_cloud_a = convert_euler_frame_to_cartesian_frame(skeleton,
                                                           euler_frame_a)
    point_cloud_b = convert_euler_frame_to_cartesian_frame(skeleton,
                                                           euler_frame_b)
    theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a, point_cloud_b, weights=weights)
    t_point_cloud_b = transform_point_cloud(point_cloud_b, theta, offset_x, offset_z)
    error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    if return_transform:
        return error, theta, offset_x, offset_z
    else:
        return error


def quat_distance(quat_a, quat_b):
    # normalize quaternion vector first
    quat_a = np.asarray(quat_a)
    quat_b = np.asarray(quat_b)
    quat_a /= np.linalg.norm(quat_a)
    quat_b /= np.linalg.norm(quat_b)
    rotmat_a = quaternion_matrix(quat_a)
    rotmat_b = quaternion_matrix(quat_b)
    diff_mat = rotmat_a - rotmat_b
    tmp = np.ravel(diff_mat)
    diff = np.linalg.norm(tmp)
    return diff


def calculate_weighted_frame_distance_quat(quat_frame_a,
                                           quat_frame_b,
                                           weights):
    assert len(quat_frame_a) == len(quat_frame_b) and \
        len(quat_frame_a) == (len(weights) - 2) * LEN_QUAT + LEN_ROOT
    diff = 0
    for i in range(len(weights) - 2):
        quat1 = quat_frame_a[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
        quat2 = quat_frame_b[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
        tmp = quat_distance(quat1, quat2)*weights[i]
        diff += tmp
    return diff


def calculate_pose_distances_from_low_dim(skeleton, mm_models, X, Y):
    """
    Converts low dimensional vectors to euler vectors and calculates the
    pose distance error by calling calculate_pose_distance
    Parameters
    ----------
    * skeleton: BVHReader
    \tContains the skeleton definition needed for the point cloud conversion

    * mm_models: Dict of MotionPrimitives
    \tContains the motion primitives for X and Y data

     * X: List
    \tList of low dimensional vectors

    * Y: List
    \tList of low dimensional vectors
    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # assert len(X) == len(Y)
    # n = len(X)
    # errors = []
    # for i in range(n):
    #     euler_frames_a = mm_models["X"].back_project(X[i]).get_motion_vector()
    #     euler_frames_b = mm_models["Y"].back_project(Y[i]).get_motion_vector()
    #     error = calculate_pose_distance(
    #         skeleton, euler_frames_a, euler_frames_b)
    #     errors.append(error)

    # return errors


def extract_root_positions(frames):
    """

    :param frames: a list of euler or quaternion frames
    :return roots_2D: a list of root position in 2D space
    """
    roots_2D = []
    for i in range(len(frames)):
        position_2D = [frames[i][0], frames[i][2]]
        roots_2D.append(position_2D)

    return roots_2D


def pose_orientation_quat(quaternion_frame, param_range=(3,7), ref_offset=[0, 0, 1, 1]):
    """Estimate pose orientation from root orientation
    """
    ref_offset = np.array(ref_offset)
    q = quaternion_frame[param_range[0]:param_range[1]]
    rotmat = quaternion_matrix(q)
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec

def pose_orientation_euler(euler_frame, ref_offset = np.array([0, 0, 1, 1])):
    '''
    only works for rocketbox skeleton
    :param euler_frame:
    :return:
    '''
    rot_angles = euler_frame[3:6]
    rot_angles_rad = np.deg2rad(rot_angles)
    rotmat = euler_matrix(rot_angles_rad[0],
                          rot_angles_rad[1],
                          rot_angles_rad[2],
                          'rxyz')
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0],0, rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


def pose_orientation_general(euler_frame, joints, skeleton, rotation_order=['Xrotation', 'Yrotation', 'Zrotation']):
    '''
    estimate pose heading direction, the position of joints define a body plane, the projection of the normal of
    the body plane is the heading direction
    :param euler_frame:
    :param joints:
    :return:
    '''
    points = []
    for joint in joints:
        points.append(skeleton.nodes[joint].get_global_position_from_euler(euler_frame, rotation_order))
    points = np.asarray(points)
    body_plane = BodyPlane(points)
    normal_vec = body_plane.normal_vector
    dir_vec = np.array([normal_vec[0], normal_vec[2]])
    return dir_vec/np.linalg.norm(dir_vec)


def pose_orientation_from_point_cloud(point_cloud, body_plane_joint_indices):
    assert len(point_cloud.shape) == 2
    points = []
    for index in body_plane_joint_indices:
        points.append(point_cloud[index])
    points = np.asarray(points)
    body_plane = BodyPlane(points)
    normal_vec = body_plane.normal_vector
    dir_vec = np.array([normal_vec[0], normal_vec[2]])
    return dir_vec/np.linalg.norm(dir_vec)


def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # rotations = []
    # for dir_vec in dir_vecs:
    #     rotations.append(Quaternion.between(dir_vec, ref_dir))
    # return rotations


def pose_orientation_game_engine_skeleton(euler_frame, skeleton):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # forward_joints = ['Game_engine', 'Root']
    # forward_points = []
    # for joint in forward_joints:
    #     forward_points.append(skeleton.nodes[joint].get_global_position_from_euler(euler_frame))
    # forward_points = np.asarray(forward_points)
    # forward_dir = forward_points[1] - forward_points[0]
    # forward_dir[1] = 0
    # return forward_dir/np.linalg.norm(forward_dir)


def get_trajectory_dir_from_2d_points(points):
    """Estimate the trajectory heading

    Parameters
    *\Points: numpy array
    Step 1: fit the points with a 2d straight line
    Step 2: estimate the direction vector from first and last point
    """
    points = np.asarray(points)
    dir_vector = points[-1] - points[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(*points.T)
    if isnan(slope):
        orientation_vec1 = np.array([0, 1])
        orientation_vec2 = np.array([0, -1])
    else:
        orientation_vec1 = np.array([slope, 1])
        orientation_vec2 = np.array([-slope, -1])
    if np.dot(
            orientation_vec1,
            dir_vector) > np.dot(
            orientation_vec2,
            dir_vector):
        orientation_vec = orientation_vec1
    else:
        orientation_vec = orientation_vec2
    orientation_vec = orientation_vec / np.linalg.norm(orientation_vec)
    return orientation_vec


def get_dir_from_2d_points(points):
    points = np.asarray(points)
    dir = points[-1] - points[2]
    dir = dir/np.linalg.norm(dir)
    return dir


def align_quaternion_frames_only_last_frame(quat_frames, prev_frames=None, transformation=None):
    """Concatenate and align quaternion frames based on previous frames or
        given transformation

    Parameters
    ----------
    * new_frames: list
         A list of quaternion frames
    * prev_frames: list
        A list of quaternion frames
   *  transformation: dict
       Contains translation and orientation in Euler angles
    Returns:
    --------
    * transformed_frames: np.ndarray
        Quaternion frames resulting from the back projection of s,
        transformed to fit to prev_frames.

    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # # find alignment transformation or use given transformation
    # if prev_frames is not None:
    #     angle, offset = fast_quat_frames_transformation(prev_frames, quat_frames)
    #     transformed_frames = transform_quaternion_frames([quat_frames[-1]],
    #                                                      [0, angle, 0],
    #                                                      offset)
    #     return transformed_frames
    # elif prev_frames is None and transformation is not None:
    #     # align frames
    #     transformed_frames = transform_quaternion_frames([quat_frames[-1]], transformation["orientation"], transformation["position"])
    #     return transformed_frames
    # else:
    #     return [quat_frames[-1]]


def align_quaternion_frames(quat_frames, prev_frames=None, transformation=None, param_range=(3,7)):
    """align quaternion frames based on previous frames or
        given transformation

    Parameters
    ----------
    * new_frames: list
         A list of quaternion frames
    * prev_frames: list
        A list of quaternion frames
   *  transformation: dict
       Contains translation and orientation in Euler angles
    Returns:
    --------
    * transformed_frames: np.ndarray
        Quaternion frames resulting from the back projection of s,
        transformed to fit to prev_frames.

    """
    # find alignment transformation or use given transformation
    if prev_frames is not None:
        angle, offset = fast_quat_frames_transformation(prev_frames, quat_frames, param_range)
        return transform_quaternion_frames(quat_frames, [0, angle, 0], offset)
    elif prev_frames is None and transformation is not None:
        # align frames
        return transform_quaternion_frames(quat_frames, transformation["orientation"], transformation["position"])
    else:
        return quat_frames


def fast_quat_frames_transformation(quaternion_frames_a,
                                    quaternion_frames_b):
    dir_vec_a = pose_orientation_quat(quaternion_frames_a[-1])
    dir_vec_b = pose_orientation_quat(quaternion_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = quaternion_frames_a[-1][0] - quaternion_frames_b[0][0]
    offset_z = quaternion_frames_a[-1][2] - quaternion_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset


def align_euler_frames(euler_frames,
                       frame_idx,
                       ref_orientation_euler):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # new_euler_frames = deepcopy(euler_frames)
    # ref_quat = euler_to_quaternion(ref_orientation_euler)
    # root_rot_angles = euler_frames[frame_idx][3:6]
    # root_rot_quat = euler_to_quaternion(root_rot_angles)
    # quat_diff = quaternion_multiply(ref_quat, quaternion_inv(root_rot_quat))
    # for euler_frame in new_euler_frames:
    #     root_trans = euler_frame[:3]
    #     new_root_trans = point_rotation_by_quaternion(root_trans, quat_diff)
    #     euler_frame[:3] = new_root_trans
    #     root_rot_angles = euler_frame[3:6]
    #     root_rot_quat = euler_to_quaternion(root_rot_angles)

    #     new_root_quat = quaternion_multiply(quat_diff, root_rot_quat)
    #     new_root_euler = quaternion_to_euler(new_root_quat)
    #     euler_frame[3:6] = new_root_euler
    # return new_euler_frames


def translate_euler_frames(euler_frames,
                           frame_idx,
                           ref_position,
                           root_joint,
                           skeleton):
    """
    translate euler frames to ref_position, tralsation is 2D on the ground
    :param euler_frames:
    :param frame_idx:
    :param ref_position:
    :return:
    """
    root_pos = skeleton.nodes[root_joint].get_global_position_from_euler(euler_frames[frame_idx])
    if len(ref_position) == 2:
        offset = np.array([ref_position[0] - root_pos[0], 0, ref_position[1] - root_pos[2]])
    elif len(ref_position) == 3:
        offset = ref_position - root_pos
        offset[1] = 0
    else:
        raise ValueError('the length of reference position is not correct! ')
    rotation = [0, 0, 0]
    return transform_euler_frames(euler_frames, rotation, offset)


def rotate_euler_frame(euler_frame,
                       ref_orientation,
                       body_plane_joints,
                       skeleton):
    forward = pose_orientation_general(euler_frame, body_plane_joints, skeleton)
    rot_angle = get_rotation_angle(ref_orientation, forward)
    translation = np.array([0, 0, 0])
    return transform_euler_frame(euler_frame, [0, rot_angle, 0], translation)


def rotate_euler_frames(euler_frames,
                        frame_idx,
                        ref_orientation,
                        body_plane_joints,
                        skeleton,
                        rotation_order=None):
    """
    Rotate a list of euler frames using the same rotation angle
    :param euler_frames: a list of euler frames to be rotated
    :param frame_idx: frame which uses to calculate rotation anlge
    :param ref_orientation: reference orientation for alignment
    :param rotation_order: define the rotation order for euler angles to calculate rotation matrix
    :global_rotation: if true, the rotation is also applied to the global position. If false, the rotation will not be applied to global position 
    :return rotated_frames: a list of rotated euler frames
    """
    forward = pose_orientation_general(euler_frames[frame_idx],
                                       body_plane_joints,
                                       skeleton)
    rot_angle = get_rotation_angle(ref_orientation, forward)
    translation = np.array([0, 0, 0])
    rotated_frames = transform_euler_frames(euler_frames,
                                            [0, rot_angle, 0],
                                            translation,
                                            rotation_order)
    return rotated_frames


def upvector_correction(filefolder,
                        frame_idx,
                        ref_upvector,
                        save_folder):
    """
    Correct the up vector of bvh files in given file folder
    :param filefolder:
    :param ref_upvector:
    :return:
    """
    if not filefolder.endswith(os.sep):
        filefolder += os.sep
    if not save_folder.endswith(os.sep):
        save_folder += os.sep
    bvhfiles = glob.glob(filefolder+'*.bvh')
    for item in bvhfiles:
        bvhreader = BVHReader(item)
        rotated_frames = rotate_euler_frames_about_x_axis(bvhreader.frames,
                                                          frame_idx,
                                                          ref_upvector)
        save_filename = save_folder + bvhreader.filename
        BVHWriter(save_filename, bvhreader, rotated_frames, bvhreader.frame_time, False)


def rotate_euler_frames_about_x_axis(euler_frames,
                                     frame_idx,
                                     ref_upvector):
    sample_upvector = pose_up_vector_euler(euler_frames[frame_idx])
    rot_angle = get_rotation_angle(ref_upvector, sample_upvector)
    translation = np.array([0, 0, 0])
    rotated_frames = transform_euler_frames(euler_frames,
                                            [rot_angle, 0, 0],
                                            translation)
    return rotated_frames


def is_vertical_pose_euler(euler_frame):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # ref_vec = np.array([1, 0, 0, 1])
    # rot_angles = np.deg2rad(euler_frame[3:6])
    # rotmat = euler_matrix(rot_angles[0],
    #                       rot_angles[1],
    #                       rot_angles[2],
    #                       'rxyz')
    # rotated_vec = np.dot(rotmat, ref_vec)
    # if rotated_vec[1] > 0.99:
    #     return True
    # else:
    #     return False


def is_vertical_pose_quat(quat_frame):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # quat_hip = np.asarray(quat_frame[3:7])
    # ref_vec = np.array([1, 0, 0, 1])
    # rotmat = quaternion_matrix(quat_hip)
    # rotated_vec = np.dot(rotmat, ref_vec)
    # if rotated_vec[1] > 0.99:
    #     return True
    # else:
    #     return False


def pose_up_vector_euler(euler_frame):
    """
    Estimate up vector of given euler frame. Assume the pose is aligned in negative Z axis, so the projection of
    3D up vector into XOY plane (Y is vertical axis) is taken into consideration.
    :param euler_frame:
    :return:
    """
    ref_offset = np.array([1, 0, 0, 1])
    rot_angles = euler_frame[3:6]
    rot_angles_rad = np.deg2rad(rot_angles)
    rotmat = euler_matrix(rot_angles_rad[0],
                          rot_angles_rad[1],
                          rot_angles_rad[2],
                          'rxyz')
    rotated_point = np.dot(rotmat, ref_offset)
    up_vec = np.array([rotated_point[0], rotated_point[1]])
    up_vec /= np.linalg.norm(up_vec)
    return up_vec


def pose_up_vector_quat(quat_frame):
    """
    Estimate up vector of given quaternion frame. Assume the pose is aligned in negative Z axis, so the projection of
    3D up vector into XOY plane (Y is vertical axis) is taken into consideration.
    :param quat_frame:
    :return:
    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # ref_offset = np.array([1, 0, 0, 1])
    # rotmat = quaternion_matrix(quat_frame[3:7])
    # rotated_point = np.dot(rotmat, ref_offset)
    # up_vec = np.array([rotated_point[0], rotated_point[2]])
    # up_vec /= np.linalg.norm(up_vec)
    # return up_vec


def quaternion_rotate_vector(q, vector):
    """ src: http://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    Args:
        q:
        vector:

    Returns:

    """
    tmp_result = quaternion_multiply(q, vector)
    return quaternion_multiply(tmp_result, quaternion_conjugate(q))[1:]


def point_rotation_by_quaternion(point, q):
    """
    Rotate a 3D point by quaternion
    :param point: [x, y, z]
    :param q: [qw, qx, qy, qz]
    :return rotated_point: [x, y, ]
    """
    r = [0, point[0], point[1], point[2]]
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    return quaternion_multiply(quaternion_multiply(q, r), q_conj)[1:]


def convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame):
    if type(euler_frame) != list:
        euler_frame = list(euler_frame)
    reduced_euler_frame = euler_frame[:3]
    for node_name in bvhreader.node_names:
        if not node_name.startswith('Bip') and 'EndSite' not in node_name:
            x_idx = bvhreader.node_channels.index((node_name, 'Xrotation'))
            reduced_euler_frame += euler_frame[x_idx: x_idx + LEN_EULER]
    return reduced_euler_frame


def convert_euler_frames_to_reduced_euler_frames(bvhreader, euler_frames):
    reduced_euler_frames = []
    for euler_frame in euler_frames:
        reduced_euler_frames.append(convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame))
    return reduced_euler_frames


def point_rotation_by_quaternion(point, q):
    """
    Rotate a 3D point by quaternion
    :param point: [x, y, z]
    :param q: [qw, qx, qy, qz]
    :return rotated_point: [x, y, ]
    """
    r = [0, point[0], point[1], point[2]]
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    return quaternion_multiply(quaternion_multiply(q, r), q_conj)[1:]
    return q/ np.linalg.norm(q)


def get_lookAt_direction(bvh_analyzer, frame_index):
    ref_vec = np.array([0, 0, 1, 0])
    global_trans = bvh_analyzer.get_global_transform("Head", frame_index)
    dir_vec = np.dot(global_trans, ref_vec)
    return np.asarray([dir_vec[0], dir_vec[2]])


def interpolate_concatenation_euler_frames(euler_frames_a, euler_frames_b, n_interpolated_frames):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # assert n_interpolated_frames%2 == 1, ("The number of intepolated frames should be odd")
    # fractions = np.linspace(0, 1, n_interpolated_frames + 2)
    # interpolated_frames = []
    # for fraction in fractions[1: -1]:
    #     new_euler_frame = quat_interpolate_euler_frames(euler_frames_a[-1],
    #                                                     euler_frames_b[0],
    #                                                     fraction)
    #     interpolated_frames.append(new_euler_frame)
    # interpolated_frames = np.asarray(interpolated_frames)
    # concatenated_frames = np.concatenate((euler_frames_a, interpolated_frames), axis=0)
    # concatenated_frames = np.concatenate((concatenated_frames, euler_frames_b), axis=0)
    # return concatenated_frames


def quat_interpolate_euler_frames(euler_frame_a, euler_frame_b, fraction):
    """
    create a new intepolated frame from two adjacent euler frames by using quaternion slerp
    The euler frame should be structured as LEN_ROOT + n_joints * LEN_EULER
    For root point, linear intepolation will be applied.
    :param euler_frame_a: numpy array
    :param euler_frame_b: numpy array
    :return:
    """
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # assert len(euler_frame_a) == len(euler_frame_b), ("The euler frames should have same length!")
    # assert (len(euler_frame_a) - LEN_ROOT)%LEN_EULER == 0, ("The lenght of euler frame is not correct!")
    # new_frame = np.zeros(len(euler_frame_a))
    # new_frame[:LEN_ROOT] = (1-fraction)*euler_frame_a[:LEN_ROOT] + fraction*euler_frame_b[:LEN_ROOT]
    # n_joints = (len(euler_frame_a) - LEN_ROOT)/LEN_EULER
    # for i in range(n_joints):
    #     joint_euler_a = euler_frame_a[i * LEN_EULER + LEN_ROOT : (i + 1) * LEN_EULER + LEN_ROOT]
    #     joint_euler_b = euler_frame_b[i * LEN_EULER + LEN_ROOT : (i + 1) * LEN_EULER + LEN_ROOT]
    #     joint_quat_a = euler_to_quaternion(joint_euler_a)
    #     joint_quat_b = euler_to_quaternion(joint_euler_b)
    #     joint_quat_new = quaternion_slerp(joint_quat_a, joint_quat_b, fraction)
    #     joint_euler_new = quaternion_to_euler(joint_quat_new)
    #     new_frame[i * LEN_EULER + LEN_ROOT : (i + 1) * LEN_EULER + LEN_ROOT] = joint_euler_new
    # return new_frame


def convert_quat_spline_to_euler_frames(quat_spline, n_frames):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # from mgrd import SkeletonMotionBVHExporter
    # bvh_exporter = SkeletonMotionBVHExporter(quat_spline)
    # # evaluation_times = np.arange(start=quat_spline.knots[0], stop=quat_spline.knots[-1], n_frames)
    # evaluation_times = np.linspace(quat_spline.knots[0], quat_spline.knots[-1], n_frames)
    # euler_frames = []
    # for frame_time in evaluation_times:
    #     euler_frames.append(bvh_exporter.generate_complete_euler_frame(frame_time))
    # return np.asarray(euler_frames)


def remove_xz_translation(cartesian_frames):
    '''

    :param cartesian_frames (ndarray): n_frames * n_joints * n_dims
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # cartesian_frames[:, 0, 0] = cartesian_frames[:, 0, 0] - cartesian_frames[0, 0, 0]
    # cartesian_frames[:, 0, 2] = cartesian_frames[:, 0, 2] - cartesian_frames[0, 0, 2]
    # return cartesian_frames


def find_rotation_angle_between_two_vector(v1, v2):
    from scipy.optimize import minimize
    '''
    positive angle means counterclockwise, negative angle means clockwise
    :return:
    '''
    def objective_function(theta, data):
        src_dir, target_dir = data
        rotmat = np.zeros((2, 2))
        rotmat[0, 0] = np.cos(theta)
        rotmat[0, 1] = -np.sin(theta)
        rotmat[1, 0] = np.sin(theta)
        rotmat[1, 1] = np.cos(theta)
        rotated_src_dir = np.dot(rotmat, src_dir)
        return np.linalg.norm(rotated_src_dir - target_dir)

    initial_guess = 0
    params = [v1, v2]
    res = minimize(objective_function, initial_guess, args=params, method='L-BFGS-B')
    return np.rad2deg(res.x[0])


def get_rotation_velocity(skeleton, euler_frames, body_plane):
    '''
    compute rotation speed around up-axis for motion in degree
    :param skeleton:
    :param euler_frames:
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # n_frames = len(euler_frames)
    # r_v = np.zeros(n_frames)
    # for i in range(n_frames-1):
    #     dir_0 = pose_orientation_general(euler_frames[i], body_plane, skeleton)
    #     dir_1 = pose_orientation_general(euler_frames[i+1], body_plane, skeleton)
    #     r_v[i+1] = find_rotation_angle_between_two_vector(dir_0, dir_1)
    # return r_v


def get_xz_translational_velocity(cartesian_frames):
    '''

    :param cartesian_frames (ndarray): n_frames * n_joints * n_dims
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # n_frames = len(cartesian_frames)
    # v = np.zeros((n_frames, 2))
    # for i in range(n_frames-1):
    #     v[i+1, 0] = cartesian_frames[i+1, 0, 0] - cartesian_frames[i, 0, 0]
    #     v[i+1, 1] = cartesian_frames[i+1, 0, 2] - cartesian_frames[i, 0, 2]
    # return v


def rotate_cartesian_frames_to_ref_dir(cartesian_frames, ref_dir, body_plane_indices, up_axis):
    '''
    rotation is assumed only about vertical axis
    :param cartesian_frames: n_frames * n_joints * 3
    :param ref_dir: 3d array
    :param body_plane_indices: 3d array (indices of three joints)
    :param up_axis: 3d array
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # n_frames, n_joints, n_dims = cartesian_frames.shape
    # forward_dir = cartesian_pose_orientation(cartesian_frames[0], body_plane_indices, up_axis)
    # axis_indice = np.where(up_axis==0)
    # angle = np.deg2rad(get_rotation_angle(forward_dir[axis_indice], ref_dir[axis_indice]))
    # rotmat = euler_matrix(0, -angle, 0, 'rxyz')
    # ones = np.ones((n_frames, n_joints, 1))
    # extended_cartesian_frames = np.concatenate((cartesian_frames, ones), axis=-1)
    # extended_cartesian_frames = np.transpose(extended_cartesian_frames, (0, 2, 1))
    # rotated_cartesian_frames = np.matmul(rotmat, extended_cartesian_frames)
    # return np.transpose(rotated_cartesian_frames, (0, 2, 1))[:, :, :-1]


def cartesian_pose_orientation(cartesian_pose, body_plane_index, up_axis):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # assert len(cartesian_pose.shape) == 2
    # points = cartesian_pose[body_plane_index, :]
    # body_plane = BodyPlane(points)
    # normal_vec = body_plane.normal_vector
    # normal_vec[np.where(up_axis == 1)] = 0  ### only consider forward direction on the ground
    # return normal_vec/np.linalg.norm(normal_vec)


def get_rotation_angles_for_vectors(dir_vecs, ref_dir, up_axis):
    '''

    :param dir_vecs: nd array N*3
    :param ref_dir: 3d array
    :param up_axis: 3d array
    :return:
    '''
    axis_indice = np.where(up_axis == 0)
    angles = []
    for i in range(len(dir_vecs)):
        angles.append(get_rotation_angle(dir_vecs[i][axis_indice], ref_dir[axis_indice]))
    return np.deg2rad(np.asarray(angles))


def rotation_cartesian_frames(cartesian_frames, angles):
    '''
    A fast version to rotate a list of cartesian frames
    :param cartesian_frames: n_frames * n_joints * 3
    :param angles: n_frames * 1
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # n_frames, n_joints, n_dims = cartesian_frames.shape
    # sin_theta = np.sin(angles)
    # cos_theta = np.cos(angles)
    # rotmat = np.array([cos_theta, np.zeros(n_frames), sin_theta, np.zeros(n_frames), np.zeros(n_frames),
    #                    np.ones(n_frames), np.zeros(n_frames), np.zeros(n_frames), -sin_theta, np.zeros(n_frames),
    #                    cos_theta, np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames),
    #                    np.ones(n_frames)]).T
    # rotmat = np.reshape(rotmat, (n_frames, 4, 4))
    # ones = np.ones((n_frames, n_joints, 1))
    # extended_cartesian_frames = np.concatenate((cartesian_frames, ones), axis=-1)
    # swapped_cartesian_frames = np.transpose(extended_cartesian_frames, (0, 2, 1))
    # rotated_cartesian_frames = np.matmul(rotmat, swapped_cartesian_frames)
    # return np.transpose(rotated_cartesian_frames, (0, 2, 1))[:, :, :-1]


def smooth_quat_frames(quat_frames):
    smoothed_quat_frames = OrderedDict()
    filenames = list(quat_frames.keys())
    smoothed_quat_frames_data = np.asarray(deepcopy(list(quat_frames.values())))
    print(('quaternion frame data shape: ', smoothed_quat_frames_data.shape))
    assert len(smoothed_quat_frames_data.shape) == 3, ('The shape of quaternion frames is not correct!')
    n_samples, n_frames, n_dims = smoothed_quat_frames_data.shape
    n_joints = int((n_dims - 3) / 4)
    for i in range(n_joints):
        ref_quat = smoothed_quat_frames_data[0, 0,
                   i * LEN_QUAT + LEN_ROOT: (i + 1) * LEN_QUAT + LEN_ROOT]
        for j in range(1, n_samples):
            test_quat = smoothed_quat_frames_data[j, 0,
                        i * LEN_QUAT + LEN_ROOT: (i + 1) * LEN_QUAT + LEN_ROOT]
            if not areQuatClose(ref_quat, test_quat):
                smoothed_quat_frames_data[j, :,
                i * LEN_QUAT + LEN_ROOT: (i + 1) * LEN_QUAT + LEN_ROOT] *= -1
    for i in range(len(filenames)):
        smoothed_quat_frames[filenames[i]] = smoothed_quat_frames_data[i]
    return smoothed_quat_frames


def print_dict_key(input_dict, level=0):
    '''
    Print key name of a dictionary recursively
    :param input_dict:
    :param level:
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # for key, value in input_dict.items():
    #     print('  ' * level + key)
    #     if isinstance(value, dict):

    #         print_dict_key(value, level+1)


def max_smoothing(x, window_size):
    '''
    smooth a binary signal by majority vote, to remove small blocks which might be caused by noise
    :param x:
    :param window_size:
    :return:
    '''
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    # if window_size % 2 == 0:
    #     window_size += 1
    # half_window = window_size // 2
    # extented_x = np.zeros(2 * half_window + len(x))
    # extented_x[:half_window] = x[0]
    # extented_x[-half_window:] = x[-1]
    # extented_x[half_window: -half_window] = x
    # new_x = np.zeros(len(x))
    # for i in range(len(x)):
    #     vs = np.average(extented_x[i: i+window_size])
    #     if vs >= 0.5:
    #         new_x[i] = 1
    #     else:
    #         new_x[i] = 0
    # return new_x


def combine_motion_clips(clips, motion_len, window_step):
    '''

    :param clips:
    :param motion_len:
    :param window_step:
    :return:
    '''
    clips = np.asarray(clips)
    n_clips, window_size, n_dims = clips.shape

    ## case 1: motion length is smaller than window_step
    if motion_len <= window_step:
        assert n_clips == 1
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    ## case 2: motion length is larger than window_step and smaller than window
    if motion_len > window_step and motion_len <= window_size:
        assert n_clips == 2
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    residue_frames = motion_len % window_step
    print('residue_frames: ', residue_frames)
    ## case 3: residue frames is smaller than window step
    if residue_frames <= window_step:
        residue_frames += window_step
        combined_frames = np.concatenate(clips[0:n_clips - 2, :window_step], axis=0)
        left_index = (window_size - residue_frames) // 2 + (window_size - residue_frames) % 2
        right_index = window_size - (window_size - residue_frames) // 2
        combined_frames = np.concatenate((combined_frames, clips[-2, left_index:right_index]), axis=0)
        return combined_frames