"""
Functions for retargeting based on the paper "Using an Intermediate Skeleton and Inverse Kinematics for Motion Retargeting"
by Monzani et al.
See: http://www.vis.uni-stuttgart.de/plain/vdl/vdl_upload/91_35_retargeting%20monzani00using.pdf
"""
import numpy as np
from ...external.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_from_euler
from .utils import normalize, find_rotation_between_vectors, to_local_cos, to_global_cos, rotate_axes, apply_additional_rotation_on_frames
from .constants import EXTRA_ROOT_NAME, GAME_ENGINE_ROOT_JOINT, ROCKETBOX_ROOT_OFFSET, OPENGL_UP_AXIS
from scipy.optimize import minimize
import math
import time





def get_dir_to_child(skeleton, name, child_name, frame, use_cache=False):

    child_pos = skeleton.nodes[child_name].get_global_position(frame, use_cache)
    global_target_dir = child_pos - skeleton.nodes[name].get_global_position(frame, True)
    global_target_dir /= np.linalg.norm(global_target_dir)
    return global_target_dir


def ik_dir_objective(q, new_skeleton, free_joint_name, targets, frame, offset):
    """ Get distance to multiple target directions similar to the Blender implementation based on FK methods
        of the Skeleton class
    """

    error = 0.0
    frame[offset: offset + 4] = q
    for target in targets:
        bone_dir = get_dir_to_child(new_skeleton, free_joint_name, target["dir_name"], frame)
        delta = bone_dir - target["dir_to_child"]
        error += np.linalg.norm(delta)
    return error


def find_rotation_using_optimization(new_skeleton, free_joint_name, targets, frame, offset, guess=None):
    if guess is None:
        guess = [1, 0, 0, 0]
    args = new_skeleton, free_joint_name, targets, frame, offset
    r = minimize(ik_dir_objective, guess, args)
    q = normalize(r.x)
    return q


def ik_dir_objective2(q, skeleton, parent_transform, targets, free_joint_offset):
    """ get distance based on precomputed parent matrices
    """
    local_m = quaternion_matrix(q)
    local_m[:3, 3] = free_joint_offset

    global_m = np.dot(parent_transform, local_m)
    free_joint_pos = global_m[:3,3]

    error = 0
    for target in targets:
        local_target_m = np.eye(4)
        local_target_m[:3, 3] = skeleton.nodes[target["dir_name"]].offset
        bone_dir = np.dot(global_m, local_target_m)[:3, 3] - free_joint_pos
        bone_dir = normalize(bone_dir)
        delta = target["dir_to_child"] - bone_dir
        error += np.linalg.norm(delta)#  np.dot(delta, delta) leads to instability. maybe try to normalize.
    return error


def find_rotation_using_optimization2(new_skeleton, free_joint_name, targets, frame, offset, guess=None):
    if guess is None:
        guess = [1, 0, 0, 0]

    parent = new_skeleton.nodes[free_joint_name].parent
    if parent is not None:
        parent_name = parent.node_name
        parent_transform = new_skeleton.nodes[parent_name].get_global_matrix(frame)

    else:
        parent_transform = np.eye(4)

    args = new_skeleton, parent_transform, targets, new_skeleton.nodes[free_joint_name].offset
    r = minimize(ik_dir_objective2, guess, args)
    q = normalize(r.x)

    return q


def find_rotation_analytically(new_skeleton, free_joint_name, target, frame):
    #find global rotation
    offset = new_skeleton.nodes[target["dir_name"]].offset
    target_dir = normalize(target["dir_to_child"])

    q = find_rotation_between_vectors(offset, target_dir)

    # bring into parent coordinate system
    pm = new_skeleton.nodes[target["dir_name"]].parent.get_global_matrix(frame)
    pm[:3,3] = [0, 0, 0]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)
    lr = np.dot(inv_pm, r)
    q = quaternion_from_matrix(lr)
    return q


def find_angle_x(v):
    # angle around y axis to rotate v to match 1,0,0
    if v[0] == 0:
        if v[1] > 0:
            return -0.5 * math.pi
        else:
            return 0.5 * math.pi
    if v[0] == 1:
        return 0
    if v[0] == -1:
        return math.pi

    alpha = math.acos(v[0])
    if v[1] > 0:
        alpha = - alpha
    return alpha

def find_rotation_analytically_from_axis(new_skeleton, free_joint_name, target, frame, offset):
    #find global orientation
    global_src_up_vec = normalize(target["global_src_up_vec"])
    global_src_x_vec = normalize(target["global_src_x_vec"])
    local_target_up_vec = [0,1,0]
    local_target_x_vec = [1,0,0]
    #find rotation between up vector of target skeleton and global src up axis
    qy = find_rotation_between_vectors(local_target_up_vec, global_src_up_vec)

    #find rotation around y axis after aligning y axis
    m = quaternion_matrix(qy)[:3,:3]
    local_target_x_vec = np.dot(m, local_target_x_vec)
    local_target_x_vec = normalize(local_target_x_vec)
    #q = q1
    qx = find_rotation_between_vectors(local_target_x_vec, global_src_x_vec)
    q = quaternion_multiply(qx,qy)
    q = to_local_cos(new_skeleton, free_joint_name, frame, q)

    #frame[offset:offset+4] = lq
    #axes = get_coordinate_system_axes(new_skeleton,free_joint_name, frame, AXES)
    #v = change_of_basis(global_src_x_vec, *axes)
    #normalize(v)
    #a = find_angle_x(v)
    #xq = quaternion_from_axis_angle(global_src_up_vec, a)
    #q = quaternion_multiply(q1,xq)
    #bring into local coordinate system of target skeleton
    #q = to_local_cos(new_skeleton, free_joint_name,frame, q)


    #m = quaternion_matrix(q1)[:3,:3]
    #x_target = np.dot(m, x_target)
    #q2 = to_local_cos(new_skeleton, free_joint_name,frame, q2)
    #
    return q


def find_rotation_analytically2_bak(new_skeleton, free_joint_name, target, frame, joint_cos_map):
    #find global orientation
    global_src_up_vec = target["global_src_up_vec"]
    global_src_x_vec = target["global_src_x_vec"]
    #local_target_up_vec = [0,1,0] #TODO get from target skeleton
    #if free_joint_name == "pelvis":
    #    local_target_x_vec = [-1, 0, 0]  # TODO get from target skeleton
    #else:
    #    local_target_x_vec = [1,0,0]  # TODO get from target skeleton

    local_target_x_vec = joint_cos_map[free_joint_name]["x"]
    local_target_up_vec = joint_cos_map[free_joint_name]["y"]
    #find rotation between up vector of target skeleton and global src up axis
    qy = find_rotation_between_vectors(local_target_up_vec, global_src_up_vec)

    #apply the rotation on the  local_target_x_vec
    m = quaternion_matrix(qy)[:3,:3]
    aligned_target_x_vec = np.dot(m, local_target_x_vec)
    aligned_target_x_vec = normalize(aligned_target_x_vec)

    # find rotation around y axis as rotation between aligned_target_x_vec and global_src_x_vec
    qx = find_rotation_between_vectors(aligned_target_x_vec, global_src_x_vec)
    #print "r", aligned_target_x_vec, global_src_x_vec, qx
    if not np.isnan(qx).any():# and free_joint_name != new_skeleton.root
        q = quaternion_multiply(qx, qy)
    else:
        q = qy
    q = to_local_cos(new_skeleton, free_joint_name, frame, q)
    return q




def get_twisting_angle2(skeleton, node_name, frame,q, joint_cos_map, dest):
    lq = to_local_cos(skeleton, node_name, frame,q)
    axes = joint_cos_map[node_name]
    local_aligned_target_axes = rotate_axes(axes, lq)
    # align aligned_axes of target with aligned local axes of src
    target = local_aligned_target_axes["x"]
    src = dest["local_src_x_vec"] # todo
    angle = math.acos(np.dot(target, src))
    #angle = math.atan2(target[0], target[2]) - math.atan2(src[2], src[0])
    e = [0,angle,0]
    lq = quaternion_from_euler(*e)
    return to_global_cos(skeleton, node_name, frame, lq)

def get_twisting_angle_local(skeleton, node_name, frame,q, joint_cos_map, local_src_x_vec):
    lq = to_local_cos(skeleton, node_name, frame,q)
    axes = joint_cos_map[node_name]
    local_aligned_target_axes = rotate_axes(axes, lq)
    #angle = get_twisting_angle(local_aligned_target_axes, local_src_x_vec)
    angle = get_angle_for_y_axis(local_aligned_target_axes["x"], local_src_x_vec, local_aligned_target_axes)

    e = [0, angle, 0]
    lq = quaternion_from_euler(*e)
    print(node_name, np.rad2deg(angle))
    return to_global_cos(skeleton, node_name, frame, lq)

def get_twisting_angle(axes, global_src_x_vec):
    angle = get_angle_for_y_axis(axes["x"], global_src_x_vec, axes)
    q = quaternion_from_axis_angle(axes["y"], angle)
    print(axes["y"], global_src_x_vec, np.degrees(angle))
    return q

def project_vector(v, x,y):
    m = [x,y]
    return np.dot(m, v)

def get_angle_for_y_axis(u, v, axes):
    """src: http://stackoverflow.com/questions/42554960/get-xyz-angles-between-vectors"""
    u = project_vector(u, axes["x"],axes["z"])
    v = project_vector(v, axes["x"],axes["z"])
    angle = math.acos(np.dot(u,v))
    return angle



def get_targets_from_motion(src_skeleton, src_frames, src_to_target_joint_map, additional_rotation_map=None):
    if additional_rotation_map is not None:
        src_frames = apply_additional_rotation_on_frames(src_skeleton.animated_joints, src_frames, additional_rotation_map)

    targets = []
    for idx in range(0, len(src_frames)):
        frame_targets = dict()
        for src_name in src_skeleton.animated_joints:
            if src_name not in list(src_to_target_joint_map.keys()):
                #print "skip1", src_name
                continue
            target_name = src_to_target_joint_map[src_name]
            frame_targets[target_name] = dict()
            frame_targets[target_name]["pos"] = src_skeleton.nodes[src_name].get_global_position(src_frames[idx])

            if len(src_skeleton.nodes[src_name].children) > -1:
                frame_targets[target_name]["targets"] = []
                for child_node in src_skeleton.nodes[src_name].children:
                    child_name = child_node.node_name

                    if child_name not in list(src_to_target_joint_map.keys()):
                        #print "skip2", src_name
                        continue
                    target = {"dir_to_child": get_dir_to_child(src_skeleton, src_name, child_name,
                                                               src_frames[idx]),
                              "dir_name": src_to_target_joint_map[child_name],
                              "src_name": src_name

                              }
                    frame_targets[target_name]["targets"].append(target)
        targets.append(frame_targets)
    return targets


def find_orientation_of_extra_root(target_skeleton, new_frame, target_root, use_optimization=True):
    extra_root_target = {"dir_name": target_root, "dir_to_child": OPENGL_UP_AXIS}
    if use_optimization:
        q = find_rotation_using_optimization(target_skeleton, EXTRA_ROOT_NAME, [extra_root_target],
                                                          new_frame, 3)
    else:
        q = find_rotation_analytically(target_skeleton, EXTRA_ROOT_NAME, extra_root_target, new_frame)
    return q


def find_rotation_of_joint(target_skeleton, free_joint_name, targets, new_frame, offset, target_root, use_optimization=True):
    if free_joint_name == target_root:
        q = find_rotation_using_optimization(target_skeleton, free_joint_name,
                                             targets,
                                             new_frame, offset)
    else:
        if use_optimization:
            q = find_rotation_using_optimization2(target_skeleton, free_joint_name,
                                                  targets,
                                                  new_frame, offset)
        else:
            q = find_rotation_analytically(target_skeleton, free_joint_name, targets[0], new_frame)

            #if free_joint_name ==  target_root:
            #    q = quaternion_multiply(q, quaternion_from_euler(*np.radians([0,180,0])))
    return q


def get_new_frames_from_direction_constraints(target_skeleton,
                                              targets, frame_range=None,
                                              target_root=GAME_ENGINE_ROOT_JOINT,
                                              src_root_offset=ROCKETBOX_ROOT_OFFSET,
                                              extra_root=True, scale_factor=1.0,
                                              use_optimization=True):

    n_params = len(target_skeleton.animated_joints) * 4 + 3

    if frame_range is None:
        frame_range = (0, len(targets))

    if extra_root:
        animated_joints = target_skeleton.animated_joints[1:]
    else:
        animated_joints = target_skeleton.animated_joints

    new_frames = []
    for frame_idx, frame_targets in enumerate(targets[frame_range[0]:frame_range[1]]):
        start = time.clock()
        target_skeleton.clear_cached_global_matrices()

        new_frame = np.zeros(n_params)
        new_frame[:3] = np.array(frame_targets[target_root]["pos"]) * scale_factor

        if extra_root:
            new_frame[:3] -= src_root_offset*scale_factor - target_skeleton.nodes[EXTRA_ROOT_NAME].offset
            new_frame[3:7] = find_orientation_of_extra_root(target_skeleton, new_frame, target_root, use_optimization)
            offset = 7
        else:
            offset = 3

        for free_joint_name in animated_joints:
            q = [1, 0, 0, 0]
            if free_joint_name in list(frame_targets.keys()) and len(frame_targets[free_joint_name]["targets"]) > 0:
                q = find_rotation_of_joint(target_skeleton, free_joint_name,
                                           frame_targets[free_joint_name]["targets"],
                                           new_frame, offset, target_root, use_optimization)
            new_frame[offset:offset + 4] = q
            offset += 4

        # apply_ik_constraints(target_skeleton, new_frame, constraints[frame_idx])#TODO
        duration = time.clock()-start
        print("processed frame", frame_range[0] + frame_idx, use_optimization, "in", duration, "seconds")
        new_frames.append(new_frame)
    return new_frames


