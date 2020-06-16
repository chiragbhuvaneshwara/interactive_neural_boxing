# encoding: UTF-8
import numpy as np
from scipy.optimize import minimize
#from python_src.morphablegraphs.animation_data.retargeting import *
from .constants import ROCKETBOX_TO_GAME_ENGINE_MAP, GAME_ENGINE_ROOT_JOINT, GAME_ENGINE_REFERENCE_POSE_EULER, EXTRA_ROOT_NAME, OPENGL_UP_AXIS, ROCKETBOX_ROOT_OFFSET
from ...external.transformations import quaternion_from_matrix, euler_matrix, quaternion_matrix, quaternion_multiply, euler_from_quaternion, quaternion_from_euler, quaternion_inverse, euler_from_matrix
from ..constants import ROTATION_TYPE_EULER, ROTATION_TYPE_QUATERNION, LEN_EULER, LEN_ROOT, LEN_QUAT


def ik_dir_euler_objective(angles, new_skeleton, free_joint_name, targets, frame, offset):
    """ get distance based on fk methods of skeleton
    """
    error = 0
    frame[offset: offset + 3] = angles
    #bone_dir = get_dir_to_child_euler(new_skeleton, free_joint_name, child_name, frame)
    for target in targets:
        bone_dir = get_dir_to_child_euler(new_skeleton, free_joint_name, target["dir_name"], frame)
        delta = np.arccos(np.dot(bone_dir, target["dir_to_child"]))
        error += np.linalg.norm(delta)
    return error


def find_rotation_euler_using_optimization(new_skeleton, free_joint_name, targets, frame, offset, guess=None):
    if guess is None:
        guess = [0, 0, 0]
    args = new_skeleton, free_joint_name, targets, frame, offset
    r = minimize(ik_dir_euler_objective, guess, args, method='BFGS')
    return r.x


def get_dir_to_child_euler(skeleton, name, child_name, frame, use_cache=False):
    child_pos = skeleton.nodes[child_name].get_global_position_from_euler(frame, use_cache)
    global_target_dir = child_pos - skeleton.nodes[name].get_global_position_from_euler(frame, use_cache)
    global_target_dir /= np.linalg.norm(global_target_dir)
    return global_target_dir


def convert_euler_frame_to_quaternion_frame(euler_frame):
    '''
    assume the euler frame contains the root translation at the beginning, and rotations are in Euler angles
    :return:
    '''
    assert len(euler_frame)%3 == 0
    n_joints = len(euler_frame)/3 - 1
    quat_frame = np.zeros(n_joints * 4 + 3)
    quat_frame[:3] = euler_frame[:3]
    for i in range(n_joints):
        rot_mat = euler_matrix(*np.deg2rad(euler_frame[i*3 + 3: (i+1)*3 + 3]), axes='rxyz')
        quat_frame[i*4 + 3:(i+1)*4 + 3] = quaternion_from_matrix(rot_mat)
    return quat_frame


def get_euler_rotation_by_name(joint_name, frame, skeleton, root_offset=3):
    assert joint_name in skeleton.animated_joints
    joint_index = skeleton.animated_joints.index(joint_name)
    return frame[joint_index * 3 + root_offset : (joint_index + 1) * 3 + root_offset]


def get_new_euler_frames_from_direction_constraints(target_skeleton,
                                                    targets,
                                                    frame_range=None,
                                                    target_root=GAME_ENGINE_ROOT_JOINT,
                                                    src_root_offset=ROCKETBOX_ROOT_OFFSET,
                                                    reference_pose=GAME_ENGINE_REFERENCE_POSE_EULER,
                                                    extra_root=True,
                                                    scale_factor=1.0):

    n_params = len(target_skeleton.animated_joints) * 3 + 3
    if frame_range is None:
        frame_range = (0, len(targets))

    new_frames = []
    for frame_idx, frame_targets in enumerate(targets[frame_range[0]:frame_range[1]]):
        target_skeleton.clear_cached_global_matrices()
        print("process", "frame " + str(frame_range[0]+frame_idx))

        new_frame = np.zeros(n_params)
        new_frame[:3] = np.array(frame_targets[target_root]["pos"]) *scale_factor

        if extra_root:
            if frame_idx == 0:
                angles = [0, 0, 0]
            else:
                angles = get_euler_rotation_by_name(EXTRA_ROOT_NAME, new_frames[frame_idx - 1], target_skeleton)
            new_frame[:3] -=  src_root_offset*scale_factor - target_skeleton.nodes[EXTRA_ROOT_NAME].offset

            targets = [{"dir_name": target_root, "dir_to_child": OPENGL_UP_AXIS}]
            new_frame[3:6] = find_rotation_euler_using_optimization(target_skeleton,
                                                                    EXTRA_ROOT_NAME,
                                                                    targets,
                                                                    new_frame, 3,
                                                                    guess=angles)
            offset = 6
        else:
            offset = 3

        for free_joint_name in target_skeleton.animated_joints[1:]:

            if frame_idx > 0:
                reference_pose = new_frames[frame_idx - 1]
            angles = get_euler_rotation_by_name(free_joint_name, reference_pose, target_skeleton)
            if free_joint_name in list(frame_targets.keys()) and len(frame_targets[free_joint_name]["targets"]) > 0:

                angles = find_rotation_euler_using_optimization(target_skeleton,
                                                                free_joint_name,
                                                                frame_targets[free_joint_name]["targets"],
                                                                new_frame, offset,
                                                                guess=angles)
            new_frame[offset:offset + 3] = angles
            offset += 3
        # apply_ik_constraints(target_skeleton, new_frame, constraints[frame_idx])#TODO
        new_frames.append(new_frame)
    return new_frames