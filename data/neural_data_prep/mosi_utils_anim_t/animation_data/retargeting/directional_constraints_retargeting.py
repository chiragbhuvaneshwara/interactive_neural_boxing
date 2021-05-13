# encoding: UTF-8
import numpy as np
from scipy.optimize import minimize
import copy
import os
import glob
from ...utilities.custom_math import euler_matrix_jac
from ..utils import get_rotation_angle, transform_euler_frame, pose_orientation_general, \
    convert_euler_frames_to_quaternion_frames, pose_orientation_from_point_cloud
from ...animation_data import LEN_ROOT, LEN_EULER, BVHReader, BVHWriter, SkeletonBuilder
from ...external.transformations import euler_matrix, euler_from_matrix                            


def get_kinematic_chain(start_joint, end_joint, skeleton):
    kinematic_chain = [end_joint]
    current_joint = skeleton.nodes[end_joint]
    while current_joint.parent is not None:

        current_joint = current_joint.parent
        kinematic_chain.append(current_joint.node_name)
        if current_joint.node_name == start_joint:
            break
    if start_joint not in kinematic_chain:
        raise ValueError("two joints are not in the same kinematic chain!")
    kinematic_chain.reverse()
    return kinematic_chain


def rotate_bone_direction(delta_angles, data):
    '''
    rotate bone direction, the bone direction is pointed from source joint to target joint
    bone direction is defined as the direction vector pointing from parent joint of target joint to target joint
    :param rotation_angles (degree):
    :param data:
    :return:
    '''

    src_joint, skeleton, target_joint, euler_frame, dofs, rotation_order = data
    assert len(delta_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    delta_rotation_angles = np.zeros(LEN_EULER)
    src_joint_index = skeleton.nodes[src_joint].quaternion_frame_index
    src_euler_angles = euler_frame[LEN_ROOT + LEN_EULER * src_joint_index: LEN_ROOT + LEN_EULER * (src_joint_index + 1)]
    if rotation_order == 'rxyz':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[2] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'rxzy':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[1] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'ryxz':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[2] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'ryzx':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[1] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'rzxy':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[0] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'rzyx':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[0] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    else:
        raise KeyError('Unknown rotation order! ')
    kinematic_chain = get_kinematic_chain(src_joint, target_joint, skeleton)
    original_rot_mat = euler_matrix(*np.deg2rad(src_euler_angles), axes=rotation_order)
    delta_rot_mat = euler_matrix(*np.deg2rad(delta_rotation_angles), axes=rotation_order)
    rot_mat = np.dot(original_rot_mat, delta_rot_mat)
    parent_joint = skeleton.nodes[src_joint].parent
    if parent_joint is not None:
        parent_global_rot = np.eye(4)
        parent_global_rot[:, :3] = parent_joint.get_global_matrix_from_euler_frame(euler_frame)[:, :3]
    else:
        parent_global_rot = np.eye(4)
    global_rot = np.dot(parent_global_rot, rot_mat)
    for joint in kinematic_chain[1: -1]:
        local_rot = np.eye(4)
        local_rot[:, :3] = skeleton.nodes[joint].get_local_matrix_from_euler(euler_frame)[:, :3]
        global_rot = np.dot(global_rot, local_rot)
    local_bone_dir = np.ones(4)
    local_bone_dir[:3] = skeleton.nodes[target_joint].offset/np.linalg.norm(skeleton.nodes[target_joint].offset)
    bone_dir = np.dot(global_rot, local_bone_dir)
    return bone_dir[:3]


def rotate_bone_direction_jac(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    src_joint, skeleton, target_joint, euler_frame, dofs, rotation_order = data
    assert len(delta_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    jac_mat = np.zeros([3, len(dofs)])
    for i in range(len(dofs)):
        jac_mat[:, i] = rotate_bone_direction_jac_one_column(delta_angles, src_joint, skeleton, target_joint,
                                                             euler_frame, dofs, dofs[i], rotation_order)
    return jac_mat


def rotate_bone_direction_jac_one_column(delta_angles, src_joint, skeleton, joint_name, euler_frame, dofs, der_axis,
                                         rotation_order):
    '''

    :param delta_angles (degrees):

    '''
    assert len(delta_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    kinematic_chain = get_kinematic_chain(src_joint, joint_name, skeleton)
    delta_rotation_angles = np.zeros(LEN_EULER)
    src_joint_index = skeleton.nodes[src_joint].quaternion_frame_index
    src_euler_angles = euler_frame[LEN_ROOT + LEN_EULER * src_joint_index: LEN_ROOT + LEN_EULER * (src_joint_index + 1)]
    if rotation_order == 'rxyz':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[2] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'rxzy':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[1] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'ryxz':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[2] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'ryzx':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[0] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[2] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'rzxy':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[0] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    elif rotation_order == 'rzyx':
        for i in range(len(dofs)):
            if dofs[i] == 'X':
                delta_rotation_angles[2] = delta_angles[i]
            elif dofs[i] == 'Y':
                delta_rotation_angles[1] = delta_angles[i]
            elif dofs[i] == 'Z':
                delta_rotation_angles[0] = delta_angles[i]
            else:
                raise KeyError('Unknown degree of free!')
    else:
        raise ValueError('Unknown rotaiton type!')

    original_rot_mat = euler_matrix(*np.deg2rad(src_euler_angles), axes=rotation_order)
    if der_axis == 'X':
        delta_rot_mat_der = euler_matrix_jac(*np.deg2rad(delta_rotation_angles), axes=rotation_order, der_axis='x')
    elif der_axis == 'Y':
        delta_rot_mat_der = euler_matrix_jac(*np.deg2rad(delta_rotation_angles), axes=rotation_order, der_axis='y')
    elif der_axis == 'Z':
        delta_rot_mat_der = euler_matrix_jac(*np.deg2rad(delta_rotation_angles), axes=rotation_order, der_axis='z')
    else:
        raise KeyError('Unknown axis')

    rot_mat = np.dot(original_rot_mat, delta_rot_mat_der)
    parent_joint = skeleton.nodes[src_joint].parent
    if parent_joint is not None:
        parent_global_rot = np.eye(4)
        parent_global_rot[:, :3] = parent_joint.get_global_matrix_from_euler_frame(euler_frame)[:, :3]
    else:
        parent_global_rot = np.eye(4)
    global_rot = np.dot(parent_global_rot, rot_mat)
    for joint in kinematic_chain[1: -1]:
        local_rot = np.eye(4)
        local_rot[:, :3] = skeleton.nodes[joint].get_local_matrix_from_euler(euler_frame)[:, :3]
        global_rot = np.dot(global_rot, local_rot)
    local_bone_dir = np.ones(4)
    local_bone_dir[:3] = skeleton.nodes[joint_name].offset/np.linalg.norm(skeleton.nodes[joint_name].offset)
    bone_dir = np.dot(global_rot, local_bone_dir)
    return bone_dir[:3]


def get_bone_direction(src_joint_angles, data):
    '''
    calculate the global direction of target joint bone using the angles of source joint, source joint should be in the
    kinematic chain to the target joint, source joint is not necessary to be adjacent with target joint

    bone direction is defined as the direction vector pointing from parent joint of target joint to target joint
    :param angles (degree): joint's parent angles
    :param params:
    :return:
    '''
    src_joint, skeleton, target_joint, euler_frame, dofs = data
    assert len(src_joint_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    src_joint_index = skeleton.nodes[src_joint].animated_joint_index
    src_euler_angles = copy.deepcopy(euler_frame[LEN_ROOT + LEN_EULER * src_joint_index:
    LEN_ROOT + LEN_EULER * (src_joint_index + 1)])
    for i in range(len(dofs)):
        if dofs[i] == 'X':
            src_euler_angles[0] = src_joint_angles[i]
        elif dofs[i] == 'Y':
            src_euler_angles[1] = src_joint_angles[i]
        elif dofs[i] == 'Z':
            src_euler_angles[2] = src_joint_angles[i]
        else:
            raise KeyError('Unknown degree of free!')
    kinematic_chain = get_kinematic_chain(src_joint, target_joint, skeleton)
    rot_mat = euler_matrix(*np.deg2rad(src_euler_angles), axes='rxyz')
    parent_joint = skeleton.nodes[src_joint].parent
    if parent_joint is not None:
        parent_global_rot = np.eye(4)
        parent_global_rot[:, :3] = parent_joint.get_global_matrix_from_euler_frame(euler_frame)[:, :3]
    else:
        parent_global_rot = np.eye(4)
    global_rot = np.dot(parent_global_rot, rot_mat)
    for joint in kinematic_chain[1: -1]:
        local_rot = np.eye(4)
        local_rot[:, :3] = skeleton.nodes[joint].get_local_matrix_from_euler(euler_frame)[:, :3]
        global_rot = np.dot(global_rot, local_rot)
    local_bone_dir = np.ones(4)
    local_bone_dir[:3] = skeleton.nodes[target_joint].offset/np.linalg.norm(skeleton.nodes[target_joint].offset)
    bone_dir = np.dot(global_rot, local_bone_dir)
    return bone_dir[:3]


def object_bone_dir_multi_joints(src_joint_angles, params):
    euler_angles, targets, src_joint, skeleton, euler_frame, dofs = params
    err = 0
    for target_joint, target_dir in targets.iteritems():
        err += object_bone_dir_multi_joints_one_target(src_joint_angles, [euler_angles, target_dir, src_joint,
                                                                          skeleton, target_joint, euler_frame, dofs])
    # print(err)
    return err


def object_bone_dir_mult_joints_jac(src_joint_angles, params):
    '''

    :param euler_angles:
    :param params:
    :return:
    '''
    jac_vec = np.zeros(len(src_joint_angles))
    targets, src_joint, skeleton, euler_frame, dofs = params
    for target_joint, target_dir in targets.items():
        jac_vec += object_bone_dir_multi_joints_one_target_jac(src_joint_angles, [target_dir, src_joint,
                                                                                  skeleton, target_joint, euler_frame,
                                                                                  dofs])
    return jac_vec


def object_bone_rotation_multi_targets(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    targets, src_joint, skeleton, euler_frame, dofs, rotation_order = data
    err = 0
    for target_joint, target_dir in targets.items():
        err += object_bone_rotation_one_target(delta_angles, [target_dir, src_joint, skeleton, target_joint,
                                                              euler_frame, dofs, rotation_order])
    # print(err)
    return err


def object_bone_rotation_multi_targets_jac(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    jac_vec = np.zeros(len(delta_angles))
    targets, src_joint, skeleton, euler_frame, dofs, rotation_order = data
    for target_joint, target_dir in targets.items():
        jac_vec += object_bone_rotation_one_target_jac(delta_angles, [target_dir, src_joint, skeleton, target_joint,
                                                                      euler_frame, dofs, rotation_order])
    return jac_vec


def object_bone_dir_multi_joints_one_target(src_joint_angles, data):
    '''

    :param src_joint_angles (radian):
    :param data:
    :return:
    '''
    target_dir, src_joint, skeleton, target_joint, euler_frame, dofs = data
    bone_dir = bone_dir_multijoints(src_joint_angles, [src_joint, skeleton, target_joint, euler_frame, dofs])
    return np.linalg.norm(bone_dir - target_dir) ** 2


def object_bone_dir_multi_joints_one_target_jac(src_joint_angles, data):
    '''

    :param src_joint_angles:
    :param data:
    :return:
    '''
    target_dir, src_joint, skeleton, target_joint, euler_frame, dofs = data
    bone_dir = bone_dir_multijoints(src_joint_angles, [src_joint, skeleton, target_joint, euler_frame, dofs])
    bone_dir_jac = bone_dir_multijoints_jac(src_joint_angles, [src_joint, skeleton, target_joint, euler_frame, dofs])
    return 2 * np.dot((bone_dir - target_dir), bone_dir_jac)


def object_bone_rotation_one_target(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    target_dir, src_joint, skeleton, target_joint, euler_frame, dofs, rotation_order = data
    bone_dir = rotate_bone_dir_multijoints(delta_angles, [src_joint, skeleton, target_joint, euler_frame, dofs,
                                                          rotation_order])
    return np.linalg.norm(bone_dir - target_dir) ** 2


def object_bone_rotation_one_target_jac(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    target_dir, src_joint, skeleton, target_joint, euler_frame, dofs, rotation_order = data
    bone_dir = rotate_bone_dir_multijoints(delta_angles, [src_joint, skeleton, target_joint, euler_frame, dofs,
                                                          rotation_order])
    bone_dir_jac = rotate_bone_dir_multijoints_jac(delta_angles, [src_joint, skeleton, target_joint, euler_frame, dofs,
                                                                  rotation_order])
    return 2 * np.dot((bone_dir - target_dir), bone_dir_jac)


def bone_dir_multijoints(src_joint_angles, data):
    joint_dir = bone_vec_multijoints(src_joint_angles, data)
    return joint_dir/np.linalg.norm(joint_dir)


def bone_dir_multijoints_jac(src_joint_angles, data):
    '''

    '''
    f = bone_vec_multijoints(src_joint_angles, data)
    f_jac = bone_vec_multijoints_jac(src_joint_angles, data)
    f = f[:, np.newaxis]
    numerator = np.linalg.norm(f) * f_jac - 0.5 * np.dot(f, np.dot(f.T, f_jac))/np.linalg.norm(f)
    denominator = np.linalg.norm(f) ** 2
    return numerator/denominator


def rotate_bone_dir_multijoints(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    joint_dir = rotate_bone_vec_multijoints(delta_angles, data)
    return joint_dir/np.linalg.norm(joint_dir)


def rotate_bone_dir_multijoints_jac(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    f = rotate_bone_vec_multijoints(delta_angles, data)
    f_jac = rotate_bone_vec_multijoints_jac(delta_angles, data)
    f = f[:, np.newaxis]
    numerator = np.linalg.norm(f) * f_jac - 0.5 * np.dot(f, np.dot(f.T, f_jac))/np.linalg.norm(f)
    denominator = np.linalg.norm(f) ** 2
    return numerator/denominator


def bone_vec_multijoints(src_joint_angles, params):
    '''
    get bone direction vector over multiple joints
    :param src_joint_angles (degree):
    :param params:
    :return:
    '''
    src_joint, skeleton, target_joint, euler_frame, dofs = params
    joint_vec = np.zeros(3)
    kinematic_chain = get_kinematic_chain(src_joint, target_joint, skeleton)
    for joint in kinematic_chain[1:]:
        bone_len = np.linalg.norm(skeleton.nodes[joint].offset)
        bone_dir = get_bone_direction(src_joint_angles, [src_joint, skeleton, joint, euler_frame, dofs])
        joint_vec += bone_len * bone_dir
    return joint_vec


def rotate_bone_vec_multijoints(delta_angles, data):
    '''

    :param delta_angles (degree):
    :param data:
    :return:
    '''
    src_joint, skeleton, target_joint, euler_frame, dofs, rotation_order = data
    joint_vec = np.zeros(3)
    kinematic_chain = get_kinematic_chain(src_joint, target_joint, skeleton)
    for joint in kinematic_chain[1:]:
        bone_len = np.linalg.norm(skeleton.nodes[joint].offset)
        if bone_len != 0:
            bone_dir = rotate_bone_direction(delta_angles, [src_joint, skeleton, joint, euler_frame, dofs, rotation_order])
            joint_vec += bone_len * bone_dir
    return joint_vec


def bone_vec_multijoints_jac(src_joint_angles, params):
    '''

    '''
    src_joint, skeleton, target_joint, euler_frame, dofs = params
    assert len(src_joint_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    jac_mat = np.zeros([3, len(dofs)])

    kinematic_chain = get_kinematic_chain(src_joint, target_joint, skeleton)
    for joint in kinematic_chain[1:]:
        bone_len = np.linalg.norm(skeleton.nodes[joint].offset)

        bone_direction_jac = get_bone_direction_jac(src_joint_angles, [src_joint, skeleton, joint, euler_frame, dofs])
        jac_mat += bone_len * bone_direction_jac
    return jac_mat


def rotate_bone_vec_multijoints_jac(delta_angles, data):
    '''

    :param delta_angles:
    :param data:
    :return:
    '''
    src_joint, skeleton, target_joint, euler_frame, dofs, rotation_order = data
    assert len(delta_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    jac_mat = np.zeros([3, len(dofs)])

    kinematic_chain = get_kinematic_chain(src_joint, target_joint, skeleton)
    for joint in kinematic_chain[1:]:
        bone_len = np.linalg.norm(skeleton.nodes[joint].offset)
        if bone_len != 0:
            bone_direction_jac = rotate_bone_direction_jac(delta_angles, [src_joint, skeleton, joint, euler_frame,
                                                                          dofs, rotation_order])
            jac_mat += bone_len * bone_direction_jac
    return jac_mat


def get_bone_direction_jac(src_joint_angles, data):
    '''

    :param angles (degree):
    :param params:
    :return:
    '''
    src_joint, skeleton, target_joint, euler_frame, dofs = data
    assert len(src_joint_angles) == len(dofs), "Number of angles do not equal to degree of freedoms"
    jac_mat = np.zeros([3, len(dofs)])
    for i in range(len(dofs)):
        jac_mat[:, i] = bone_dir_jac_one_column(src_joint_angles, src_joint, skeleton, target_joint, euler_frame, dofs,
                                                dofs[i])
    return jac_mat


def bone_dir_jac_one_column(angles, src_joint, skeleton, joint_name, euler_frame, dofs, der_axis):
    '''
    calculate partial derivative for one axis
    :param angles:
    :param params:
    :param der_axis:
    :return:
    '''
    kinematic_chain = get_kinematic_chain(src_joint, joint_name, skeleton)
    src_joint_index = skeleton.nodes[src_joint].animated_joint_index
    src_euler_angles = copy.deepcopy(euler_frame[LEN_ROOT + LEN_EULER * src_joint_index:
    LEN_ROOT + LEN_EULER * (src_joint_index + 1)])
    for i in range(len(dofs)):
        if dofs[i] == 'X':
            src_euler_angles[0] = angles[i]
        elif dofs[i] == 'Y':
            src_euler_angles[1] = angles[i]
        elif dofs[i] == 'Z':
            src_euler_angles[2] = angles[i]
        else:
            raise KeyError('Unknown degree of free!')
    if der_axis == 'X':
        rot_mat = euler_matrix_jac(*np.deg2rad(src_euler_angles), axes='rxyz', der_axis='x')
    elif der_axis == 'Y':
        rot_mat = euler_matrix_jac(*np.deg2rad(src_euler_angles), axes='rxyz', der_axis='y')
    elif der_axis == 'Z':
        rot_mat = euler_matrix_jac(*np.deg2rad(src_euler_angles), axes='rxyz', der_axis='z')
    else:
        raise KeyError('Unknown axis')
    parent_joint = skeleton.nodes[src_joint].parent
    # grandparent_joint = parent_joint.parent
    if parent_joint is not None:
        parent_global_rot = np.eye(4)
        parent_global_rot[:, :3] = parent_joint.get_global_matrix_from_euler_frame(euler_frame)[:, :3]
    else:
        parent_global_rot = np.eye(4)
    global_rot = np.dot(parent_global_rot, rot_mat)
    for joint in kinematic_chain[1: -1]:
        local_rot = np.eye(4)
        local_rot[:, :3] = skeleton.nodes[joint].get_local_matrix_from_euler(euler_frame)[:, :3]
        global_rot = np.dot(global_rot, local_rot)
    local_bone_dir = np.ones(4)
    local_bone_dir[:3] = skeleton.nodes[joint_name].offset/np.linalg.norm(skeleton.nodes[joint_name].offset)
    bone_dir = np.dot(global_rot, local_bone_dir)
    return bone_dir[:3]


def estimate_scale_factor(src_rest_pose, root_joint, target_bvh, joint_mapping):
    '''
    find the ratio between the longest dimension of source skeleton and target skeleton in Cartesian space for mapped
    joints (the joint which are not in mapped joints are ignored)
    Steps:
    1. retarget the rest pose of source skeleton to target skeleton
    2. convert both rest poses into Cartesian space for mapped joints
    3. find the ratio of longest dimensions between source pose and target pose
    :return:
    '''
    src_bvhreader = BVHReader(src_rest_pose)
    src_skeleton = SkeletonBuilder().load_from_bvh(src_bvhreader)

    target_bvhreader = BVHReader(target_bvh)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)

    target = create_direction_constraints(joint_mapping, src_skeleton, src_bvhreader.frames[0])
    new_frame = copy.deepcopy(target_bvhreader.frames[0])
    retarget_motion(root_joint, target, target_skeleton, new_frame)
    # BVHWriter(r'E:\tmp\path_data\result.bvh', target_skeleton, [new_frame], target_skeleton.frame_time,
    #           is_quaternion=False)
    n_joints = len(joint_mapping)
    src_cartesian_frame = np.zeros([n_joints, 3])
    target_cartesian_frame = np.zeros([n_joints, 3])
    i = 0
    for key, value in joint_mapping.items():
        src_cartesian_frame[i*3: (i+1)*3] = src_skeleton.nodes[key].get_global_position_from_euler(src_bvhreader.frames[0])
        target_cartesian_frame[i*3: (i+1)*3] = target_skeleton.nodes[value].get_global_position_from_euler(new_frame)
        i += 1
    src_maximum = np.amax(src_cartesian_frame, axis=0)
    src_minimum = np.amin(src_cartesian_frame, axis=0)
    target_maximum = np.amax(target_cartesian_frame, axis=0)
    target_minimum = np.amin(target_cartesian_frame, axis=0)
    src_max_diff = np.max(src_maximum - src_minimum)
    target_max_diff = np.max(target_maximum - target_minimum)
    scale = target_max_diff/src_max_diff
    return scale


def retarget_single_joint(joint_name, target, target_skeleton, ref_euler_frame, joint_dof=None):
    '''

    :param joint_name:
    :param target dic: {src_joint: {target_joint: dir_vector} }
    :param target_skeleton:
    :param ref_euler_frame:
    :param joint_dof (list):
    :return:
    '''
    # print(joint_name)
    euler_index = target_skeleton.nodes[joint_name].animated_joint_index
    if joint_dof is not None:
        indices = []
        for i in range(len(joint_dof)):
            if joint_dof[i] == 'X':
                indices.append(0)
            elif joint_dof[i] == 'Y':
                indices.append(1)
            elif joint_dof[i] == 'Z':
                indices.append(2)
            else:
                raise KeyError('Unknown type of dofs')
    else:
        joint_dof = ['X', 'Y', 'Z']
        indices = [0, 1, 2]
    initial_guess = np.zeros(len(joint_dof))
    params = [target, joint_name, target_skeleton, ref_euler_frame, joint_dof]
    res = minimize(object_bone_rotation_multi_targets, initial_guess, args=params, method='L-BFGS-B',
                   jac=object_bone_rotation_multi_targets_jac)
    rotation_angles = np.zeros(LEN_EULER)
    rotation_angles[indices] = res.x
    joint_angles = ref_euler_frame[LEN_ROOT + LEN_EULER * euler_index: LEN_ROOT + (euler_index + 1) * LEN_EULER]
    rotmat1 = euler_matrix(*np.deg2rad(joint_angles), axes='rxyz')
    rotmat2 = euler_matrix(*np.deg2rad(rotation_angles), axes='rxyz')
    rotmat = np.dot(rotmat1, rotmat2)
    new_angles = euler_from_matrix(rotmat, axes='rxyz')
    ref_euler_frame[LEN_ROOT + LEN_EULER * euler_index: LEN_ROOT + (euler_index + 1) * LEN_EULER] = np.rad2deg(new_angles)


def recursive_retarget_motion(targets, target_skeleton, ref_euler_frame, joints_dofs={}, joint_name=None):
    '''
    traverse the kinematic chain starting from given joint
    :param targets:
    :param target_skeleton:
    :param ref_euler_frame:
    :param joints_dofs:
    :return:
    '''
    if joint_name is None:
        joint_name = target_skeleton.root
    if joint_name in targets.keys():
        retarget_single_joint(joint_name, targets[joint_name], target_skeleton, ref_euler_frame,
                              joints_dofs[joint_name])
    for child in target_skeleton.nodes[joint_name].children:
        recursive_retarget_motion(targets, target_skeleton, ref_euler_frame, joints_dofs, joint_name=child.node_name)


def retarget_motion(joint_name, targets, target_skeleton, ref_euler_frame, joints_dofs=None):
    '''
    recursively optimization joint orientation

    optimize the current joint rotation based on taregets
    (recursively search using source skeleton, all joints in source skeleton are targeted to target skeleton, but it is
    not necessary for joints in target skeleton be targeted)
    call retarget_motion for each child joint

    targets shouold only contain joints from target skeleton: the target direction from target joints

    :param joint_name:
    :param targets:
    :param skeleton:
    :return:
    '''
    # print("joint name: ", joint_name)
    if joints_dofs is None:
        joints_dofs = {}
    euler_index = target_skeleton.nodes[joint_name].quaternion_frame_index
    # print("euler index: ", euler_index)
    rotation_order = target_skeleton.nodes[joint_name].rotation_order
    assert rotation_order != []
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            rotation_axes = 'rxyz'
            if joint_name in joints_dofs.keys():
                joint_dofs = joints_dofs[joint_name]
                indices = []
                for i in range(len(joint_dofs)):
                    if joint_dofs[i] == 'X':
                        indices.append(0)
                    elif joint_dofs[i] == 'Y':
                        indices.append(1)
                    elif joint_dofs[i] == 'Z':
                        indices.append(2)
                    else:
                        raise KeyError('Unknown type of dofs')
            else:
                joint_dofs = ['X', 'Y', 'Z']
                indices = [0, 1, 2]
        elif rotation_order[1] == 'Zrotation':
            rotation_axes = 'rxzy'
            if joint_name in joints_dofs.keys():
                joint_dofs = joints_dofs[joint_name]
                indices = []
                for i in range(len(joint_dofs)):
                    if joint_dofs[i] == 'X':
                        indices.append(0)
                    elif joint_dofs[i] == 'Y':
                        indices.append(2)
                    elif joint_dofs[i] == 'Z':
                        indices.append(1)
                    else:
                        raise KeyError('Unknown type of dofs')
            else:
                joint_dofs = ['X', 'Z', 'Y']
                indices = [0, 1, 2]
        else:
            raise ValueError('Unknown Rotation Type!')
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            rotation_axes = 'ryxz'
            if joint_name in joints_dofs.keys():
                joint_dofs = joints_dofs[joint_name]
                indices = []
                for i in range(len(joint_dofs)):
                    if joint_dofs[i] == 'X':
                        indices.append(1)
                    elif joint_dofs[i] == 'Y':
                        indices.append(0)
                    elif joint_dofs[i] == 'Z':
                        indices.append(2)
                    else:
                        raise KeyError('Unknown type of dofs')
            else:
                joint_dofs = ['Y', 'X', 'Z']
                indices = [0, 1, 2]
        elif rotation_order[1] == 'Zrotation':
            rotation_axes = 'ryzx'
            if joint_name in joints_dofs.keys():
                joint_dofs = joints_dofs[joint_name]
                indices = []
                for i in range(len(joint_dofs)):
                    if joint_dofs[i] == 'X':
                        indices.append(2)
                    elif joint_dofs[i] == 'Y':
                        indices.append(0)
                    elif joint_dofs[i] == 'Z':
                        indices.append(1)
                    else:
                        raise KeyError('Unknown type of dofs')
            else:
                joint_dofs = ['Y', 'Z', 'X']
                indices = [0, 1, 2]
        else:
            raise ValueError('Unknown Rotation Type!')
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            rotation_axes = 'rzxy'
            if joint_name in joints_dofs.keys():
                joint_dofs = joints_dofs[joint_name]
                indices = []
                for i in range(len(joint_dofs)):
                    if joint_dofs[i] == 'X':
                        indices.append(1)
                    elif joint_dofs[i] == 'Y':
                        indices.append(2)
                    elif joint_dofs[i] == 'Z':
                        indices.append(0)
                    else:
                        raise KeyError('Unknown type of dofs')
            else:
                joint_dofs = ['Z', 'X', 'Y']
                indices = [0, 1, 2]
        elif rotation_order[1] == 'Yrotation':
            rotation_axes = 'rzyx'
            if joint_name in joints_dofs.keys():
                joint_dofs = joints_dofs[joint_name]
                indices = []
                for i in range(len(joint_dofs)):
                    if joint_dofs[i] == 'X':
                        indices.append(2)
                    elif joint_dofs[i] == 'Y':
                        indices.append(1)
                    elif joint_dofs[i] == 'Z':
                        indices.append(0)
                    else:
                        raise KeyError('Unknown type of dofs')
            else:
                joint_dofs = ['Z', 'Y', 'X']
                indices = [0, 1, 2]
        else:
            raise ValueError('Unknown Rotation Type!')
    else:
        raise ValueError('Unknown Rotation Type!')
    indices = np.asarray(indices)
    # pre_angles = ref_euler_frame[LEN_ROOT + LEN_EULER * euler_index + indices]
    # assert len(pre_angles) == len(joint_dofs)
    initial_guess = np.zeros(len(joint_dofs))
    params = [targets[joint_name], joint_name, target_skeleton, ref_euler_frame, joint_dofs, rotation_axes]

    # res = minimize(object_bone_dir_multi_joints, pre_angles, args=params, method='L-BFGS-B',
    #                jac=object_bone_dir_mult_joints_jac)
    # # print('optimal angles: ', res.x)
    # ref_euler_frame[LEN_ROOT + LEN_EULER * euler_index + indices] = res.x

    res = minimize(object_bone_rotation_multi_targets, initial_guess, args=params, method='L-BFGS-B',
                   jac=object_bone_rotation_multi_targets_jac)
    rotation_angles = np.zeros(LEN_EULER)
    rotation_angles[indices] = res.x
    joint_angles = ref_euler_frame[LEN_ROOT + LEN_EULER * euler_index: LEN_ROOT + (euler_index + 1) * LEN_EULER]

    rotmat1 = euler_matrix(*np.deg2rad(joint_angles), axes=rotation_axes)
    rotmat2 = euler_matrix(*np.deg2rad(rotation_angles), axes=rotation_axes)
    rotmat = np.dot(rotmat1, rotmat2)
    new_angles = euler_from_matrix(rotmat, axes=rotation_axes)
    ref_euler_frame[LEN_ROOT + LEN_EULER * euler_index: LEN_ROOT + (euler_index + 1) * LEN_EULER] = np.rad2deg(new_angles)
    for child in targets[joint_name].keys():
        if child in targets.keys():
            retarget_motion(child, targets, target_skeleton, ref_euler_frame, joints_dofs)


def create_direction_constraints(joint_map, src_skeleton, euler_frame, body_plane=None, constrained_joints=None,
                                 prev_frame=None):
    """
    
    Arguments:
        joint_map {dict} -- joint name mapping from src skeleton to target skeleton
        src_skeleton {Skeleton} -- Skeleton model for the source data
        euler_frame {numpy.array} -- euler angles
    
    Keyword Arguments:
        body_plane {[type]} -- [description] (default: {None})
        constrained_joints {[type]} -- [description] (default: {None})
        prev_frame {[type]} -- [description] (default: {None})
    
    Returns:
        [type] -- [description]
    """   
    targets = {}
    if body_plane is None:
        pose_dir = None
    else:
        pose_dir = pose_orientation_general(euler_frame, body_plane, src_skeleton)
    targets['pose_dir'] = pose_dir
    if constrained_joints is not None:
        targets['constrains'] = {}
    for joint in joint_map.keys():
        create_direction_constraints_recursively(joint, targets, src_skeleton, euler_frame, joint_map)
    return targets


def create_direction_constrains_recursively_from_point_cloud(joint_name, targets, src_skeleton, point_cloud, JOINT_MAP):
    '''
    for given joint, first check it is in the JOINT_MAP or not.
    for all the joint's children, if the child is in JOINT_MAP table, create a direction target from

    recursively search the predecessors until one of the predecessor is in MAP_joint
    :param joint_name: str
    :param targets:
    :param src_skeleton:
    :return:
    '''
    src_joint = src_skeleton.nodes[joint_name].parent
    assert len(point_cloud.shape) == 2
    while src_joint is not None:
        if src_joint.node_name in JOINT_MAP.keys():
            src_pos = point_cloud[src_joint.index]
            joint_pos = point_cloud[src_skeleton.nodes[joint_name].index]

            assert np.linalg.norm(joint_pos - src_pos)> 1e-6, ('Bone direction should not be zero!')

            bone_dir = (joint_pos - src_pos)/np.linalg.norm(joint_pos - src_pos)
            if JOINT_MAP[src_joint.node_name] not in targets.keys():
                targets[JOINT_MAP[src_joint.node_name]] = {JOINT_MAP[joint_name]: bone_dir}
            else:
                targets[JOINT_MAP[src_joint.node_name]][JOINT_MAP[joint_name]] = bone_dir
            break
        src_joint = src_joint.parent


def create_direction_constraints_from_point_cloud(joint_map, skeleton, point_cloud, body_plane):
    '''
    create bone direction constraints for one frame from point cloud data
    :param joint_map: recursively scan skeleton hierachiy
    :param skeleton:
    :param point_cloud: n_joints * 3
    :param body_plane:
    :return:
    '''
    targets = {}
    if body_plane is None:
        pose_dir = None
    else:
        body_plane_indices = [skeleton.animated_joints.index(joint) for joint in body_plane]
        pose_dir = pose_orientation_from_point_cloud(point_cloud, body_plane_indices)
    targets['pose_dir'] = pose_dir
    for joint_name in joint_map.keys():
        create_direction_constrains_recursively_from_point_cloud(joint_name, targets, skeleton, point_cloud, joint_map)
    return targets


def create_direction_constraints_recursively(joint, targets, src_skeleton, euler_frame, JOINT_MAP):
    '''
    for given joint, first check it is in the JOINT_MAP or not.
    for all the joint's children, if the child is in JOINT_MAP table, create a direction target from

    recursively search the predecessors until one of the predecessor is in MAP_joint
    :param joint:
    :param targets:
    :param src_skeleton:
    :return:
    '''
    src_joint = src_skeleton.nodes[joint].parent
    while src_joint is not None:
        # print(src_joint.node_name)
        if src_joint.node_name in JOINT_MAP.keys():
            src_pos = src_joint.get_global_position_from_euler(euler_frame)
            joint_pos = src_skeleton.nodes[joint].get_global_position_from_euler(euler_frame)
            # print(src_skeleton.nodes[joint].node_name)

            assert np.linalg.norm(joint_pos - src_pos) > 1e-6, ('Bone direction should not be zero!')

            bone_dir = (joint_pos - src_pos)/np.linalg.norm(joint_pos - src_pos)
            if JOINT_MAP[src_joint.node_name] not in targets.keys():
                targets[JOINT_MAP[src_joint.node_name]] = {JOINT_MAP[joint]: bone_dir}
            else:
                targets[JOINT_MAP[src_joint.node_name]][JOINT_MAP[joint]] = bone_dir
            break
        else:
            src_joint = src_joint.parent


def align_ref_frame(euler_frame, target_dir, skeleton, body_plane=None, rotation_order=['Xrotation', 'Yrotation', 'Zrotation']):
    '''

    :param euler_frame:
    :param target_dir: 2D vector
    :return:
    '''
    if body_plane is None:
        pose_dir = get_game_engine_skeleton_pose_dir(euler_frame, skeleton)
    else:
        pose_dir = pose_orientation_general(euler_frame, body_plane, skeleton, rotation_order=rotation_order)
    rotation_angle = get_rotation_angle(target_dir, pose_dir)
    rotated_frame = transform_euler_frame(euler_frame,
                                          [0, rotation_angle, 0],
                                          [0, 0, 0],
                                          rotation_order=rotation_order)
    return rotated_frame


def get_game_engine_skeleton_pose_dir(euler_frame, skeleton):
    game_eigine_pos = skeleton.nodes['Game_engine'].get_global_position_from_euler(euler_frame)
    root_pos = skeleton.nodes['Root'].get_global_position_from_euler(euler_frame)
    # print('game engine: ', game_eigine_pos)
    # print('root: ', root_pos)
    dir = game_eigine_pos - root_pos
    dir_2d = np.array([dir[0], dir[2]])
    return dir_2d/np.linalg.norm(dir_2d)


def retarget_folder(src_folder, ref_file, save_folder, joint_mapping, joints_dofs=None,
                    root_joint=None, src_body_plane=None, target_body_plane=None, src_rest_pose=None):
    '''
    apply directional retargeting to a folder consisting of bvh files
    :param src_folder:
    :param ref_file:
    :param save_folder:
    :param joint_mapping:
    :param root_joint:
    :param src_rest_pose:
    :return:
    '''
    print(src_folder)
    ref_bvhreader = BVHReader(ref_file)
    ref_skeleton = SkeletonBuilder().load_from_bvh(ref_bvhreader)
    bvhfiles = glob.glob(os.path.join(src_folder, '*.bvh'))
    if bvhfiles == []:
        return
    if src_rest_pose is None: ## if there is no rest pose file, randomly take one from input files
        src_rest_pose = bvhfiles[0]
    skeleton_scale_factor = estimate_scale_factor(src_rest_pose, root_joint, ref_file, joint_mapping)
    for bvhfile in bvhfiles:
        bvhreader = BVHReader(bvhfile)
        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        filename = os.path.split(bvhfile)[-1]
        print(filename)
        targets = []
        output_frames = []
        n_frames = len(bvhreader.frames)
        # create constraints for each frame
        for i in range(n_frames):
            if i > 0:
                targets.append(
                    create_direction_constraints(joint_mapping, skeleton, bvhreader.frames[i], src_body_plane,
                                                 bvhreader.frames[i-1]))
            else:
                targets.append(
                    create_direction_constraints(joint_mapping, skeleton, bvhreader.frames[i], src_body_plane))
        # retarget motion
        for i in range(n_frames):
            pose_dir = targets[i]['pose_dir']
            if i == 0:
                new_frame = copy.deepcopy(ref_bvhreader.frames[0]) ## create default pose
                ref_frame = align_ref_frame(new_frame, pose_dir, ref_skeleton, body_plane=target_body_plane)  ## align the default pose to the source
                                                                                ## pose direction
            else:
                new_frame = copy.deepcopy(output_frames[i-1])  ## initialize the default pose using the previous frame
                ref_frame = align_ref_frame(new_frame, pose_dir, ref_skeleton, body_plane=target_body_plane)
            ref_frame[:3] = bvhreader.frames[i][:3] * skeleton_scale_factor
            retarget_motion(root_joint, targets[i], ref_skeleton, ref_frame, joints_dofs)
            output_frames.append(ref_frame)
        BVHWriter(os.path.join(save_folder, filename), ref_skeleton, output_frames, ref_skeleton.frame_time,
                  is_quaternion=False)


def retarget_single_motion(input_file, ref_file, rest_pose, save_dir, root_joint, src_body_plane, target_body_plane,
                           joint_mapping, joints_dofs=None, constrained_joints=None):
    '''

    :param input_file: contains source motion
    :param ref_file: contains target skeleton
    :param rest_pose: define the rest post, default setting can be the same as input file
    :return:
    '''
    ref_bvhreader = BVHReader(ref_file)
    ref_skeleton = SkeletonBuilder().load_from_bvh(ref_bvhreader)
    bvhreader = BVHReader(input_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_scale_factor = estimate_scale_factor(rest_pose, root_joint, ref_file, joint_mapping)
    print('skeleton scale factor: ', skeleton_scale_factor)
    # skeleton_scale_factor = 0.05

    out_frames = []
    targets = []
    n_frames = len(bvhreader.frames)
    for i in range(n_frames):
        if i > 0:
            targets.append(
                create_direction_constraints(joint_mapping, skeleton, bvhreader.frames[i], src_body_plane,
                                             constrained_joints, bvhreader.frames[i - 1]))
        else:
            targets.append(
                create_direction_constraints(joint_mapping, skeleton, bvhreader.frames[i], src_body_plane,
                                             constrained_joints))
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        print(i)
        if i == 0:
            new_frame = ref_bvhreader.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, ref_skeleton, target_body_plane)
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, ref_skeleton, target_body_plane)
            ref_frame[:3] = (bvhreader.frames[i][:3] - bvhreader.frames[i-1][:3]) * skeleton_scale_factor + out_frames[i-1][:3]
        retarget_motion(root_joint, targets[i], ref_skeleton, ref_frame, joints_dofs)
        out_frames.append(ref_frame)
    ## apply foot IK
    # skeleton_model = GAME_ENGINE_SKELETON_MODEL
    # ik_chain = skeleton_model['ik_chains']
    # quat_frames = np.array(convert_euler_frames_to_quaternion_frames(ref_bvhreader, out_frames))
    # new_quat_frames = []
    # for i in range(n_frames-1):
    #     new_quat_frame = None
    #
    #     for joint in constrained_joints:
    #         prev_target_pos = ref_skeleton.nodes[joint_mapping[joint]].get_global_position(quat_frames[i])
    #         ik = AnalyticalLimbIK.init_from_dict(ref_skeleton, joint_mapping[joint], ik_chain[joint_mapping[joint]])
    #         target_pos = targets[i+1]['constrains'][joint] * skeleton_scale_factor + prev_target_pos
    #         # print('target position: ', target_pos)
    #         # print('origin position: ', ref_skeleton.nodes[joint_mapping[joint]].get_global_position(quat_frames[i+1]))
    #     #     if new_quat_frame is not None:
    #     #         new_quat_frame = ik.apply2(new_quat_frame, target_pos)
    #     #     else:
    #     #         new_quat_frame = ik.apply2(quat_frames[i], target_pos)
    #     # new_quat_frames.append(new_quat_frame)
    #         quat_frames[i+1] = ik.apply2(quat_frames[i+1], target_pos)
    # new_quat_frames = quat_frames


    # filename = os.path.split(input_file)[-1]
    # BVHWriter(os.path.join(save_dir, filename), ref_skeleton, quat_frames, skeleton.frame_time,
    #           is_quaternion=True)
    filename = os.path.split(input_file)[-1]
    BVHWriter(os.path.join(save_dir, filename), ref_skeleton, out_frames, skeleton.frame_time,
              is_quaternion=False)
