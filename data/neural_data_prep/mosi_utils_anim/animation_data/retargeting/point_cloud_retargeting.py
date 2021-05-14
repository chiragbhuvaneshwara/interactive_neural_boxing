"""
Functions for retargeting based on the paper "Using an Intermediate Skeleton and Inverse Kinematics for Motion Retargeting"
by Monzani et al.
See: http://www.vis.uni-stuttgart.de/plain/vdl/vdl_upload/91_35_retargeting%20monzani00using.pdf
"""
import numpy as np
import math
from .constants import OPENGL_UP_AXIS, GAME_ENGINE_SPINE_OFFSET_LIST
from .utils import normalize, align_axis, find_rotation_between_vectors, align_root_translation, to_local_cos, get_quaternion_rotation_by_name, apply_additional_rotation_on_frames, project_vector_on_axis, quaternion_from_vector_to_vector
from ...external.transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_from_matrix
from ..skeleton_models import JOINT_CHILD_MAP
from .analytical import create_local_cos_map_from_skeleton_axes_with_map, apply_manual_fixes

JOINT_CHILD_MAP = dict()
JOINT_CHILD_MAP["root"] = "pelvis"
JOINT_CHILD_MAP["pelvis"] = "spine_2"
JOINT_CHILD_MAP["spine_2"] = "neck"
JOINT_CHILD_MAP["neck"] = "head"
JOINT_CHILD_MAP["left_clavicle"] = "left_shoulder"
JOINT_CHILD_MAP["left_shoulder"] = "left_elbow"
JOINT_CHILD_MAP["left_elbow"] = "left_wrist"
JOINT_CHILD_MAP["left_wrist"] = "left_finger"
JOINT_CHILD_MAP["right_clavicle"] = "right_shoulder"
JOINT_CHILD_MAP["right_shoulder"] = "right_elbow"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_wrist"] = "right_finger"
JOINT_CHILD_MAP["left_hip"] = "left_knee"
JOINT_CHILD_MAP["left_knee"] = "left_ankle"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_hip"] = "right_knee"
JOINT_CHILD_MAP["right_knee"] = "right_ankle"
JOINT_CHILD_MAP["left_ankle"] = "left_toe"
JOINT_CHILD_MAP["right_ankle"] = "right_toe"



def get_quaternion_to_axis(skeleton, joint_a, joint_b, axis):
    ident_f = skeleton.identity_frame
    ap = skeleton.nodes[joint_a].get_global_position(ident_f)
    bp = skeleton.nodes[joint_b].get_global_position(ident_f)
    delta = bp - ap
    delta /= np.linalg.norm(delta)
    return quaternion_from_vector_to_vector(axis, delta)


def rotate_axes(cos, q):
    m = quaternion_matrix(q)[:3, :3]
    for key, a in list(cos.items()):
        cos[key] = np.dot(m, a)
        cos[key] = normalize(cos[key])
    return cos


def get_child_joint(skeleton_model, inv_joint_map, node_name, src_children_map):
    """ Warning output is random if there are more than one child joints
        and the value is not specified in the JOINT_CHILD_MAP """
    child_name = None
    if node_name in src_children_map and len(src_children_map[node_name]) > 0:
        child_name = src_children_map[node_name][-1]
    if node_name in inv_joint_map:
        joint_name = inv_joint_map[node_name]
        while joint_name in JOINT_CHILD_MAP:
            _child_joint_name = JOINT_CHILD_MAP[joint_name]

            # check if child joint is mapped
            joint_key = None
            if _child_joint_name in skeleton_model["joints"]:
                joint_key = skeleton_model["joints"][_child_joint_name]

            if joint_key is not None: # return child joint
                child_name = joint_key
                return child_name
            else: #keep traversing until end of child map is reached
                if _child_joint_name in JOINT_CHILD_MAP:
                    joint_name = JOINT_CHILD_MAP[_child_joint_name]
                    #print(joint_name)
                else:
                    break
    return child_name


def align_root_joint(new_skeleton, free_joint_name, axes, global_src_up_vec, global_src_x_vec,joint_cos_map, max_iter_count=10):
    # handle special case for the root joint
    # apply only the y axis rotation of the Hip to the Game_engine node
    q = [1, 0, 0, 0]
    #apply first time
    qx, axes = align_axis(axes, "x", global_src_x_vec)  # first find rotation to align x axis
    q = quaternion_multiply(qx, q)
    q = normalize(q)

    qy, axes = align_axis(axes, "y", global_src_up_vec)  # then add a rotation to let the y axis point up
    q = quaternion_multiply(qy, q)
    q = normalize(q)

    #apply second time
    qx, axes = align_axis(axes, "x", global_src_x_vec)  # first find rotation to align x axis
    q = quaternion_multiply(qx, q)
    q = normalize(q)
    qy, axes = align_axis(axes, "y", global_src_up_vec)  # then add a rotation to let the y axis point up
    q = quaternion_multiply(qy, q)
    q = normalize(q)

    # print("handle special case for pelvis")
    # handle special case of applying the x axis rotation of the Hip to the pelvis
    node = new_skeleton.nodes[free_joint_name]
    t_pose_global_m = node.get_global_matrix(new_skeleton.reference_frame)[:3, :3]
    global_original = np.dot(t_pose_global_m, joint_cos_map[free_joint_name]["y"])
    global_original = normalize(global_original)
    qoffset = find_rotation_between_vectors(OPENGL_UP_AXIS, global_original)
    q = quaternion_multiply(q, qoffset)
    q = normalize(q)
    return q


def align_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec, global_src_x_vec, joint_cos_map):
    # first align the bone vectors
    q = [1, 0, 0, 0]
    qy, axes = align_axis(local_target_axes, "y", global_src_up_vec)
    q = quaternion_multiply(qy, q)
    #joint_map = new_skeleton.skeleton_model["joints"]
    q = normalize(q)
    if free_joint_name == "pelvis":
        print("handle special case for pelvis")
        # handle special case of applying the x axis rotation of the Hip to the pelvis
        node = new_skeleton.nodes[free_joint_name]
        t_pose_global_m = node.get_global_matrix(new_skeleton.reference_frame)[:3, :3]
        global_original = np.dot(t_pose_global_m, joint_cos_map[free_joint_name]["y"])
        global_original = normalize(global_original)
        qoffset = find_rotation_between_vectors(OPENGL_UP_AXIS, global_original)
        q = quaternion_multiply(qoffset, q)
        q = normalize(q)

    # then align the twisting angles
    if global_src_x_vec is not None:
        #old_x = np.array(axes["x"])
        qx, axes = align_axis(axes, "x", global_src_x_vec)
        q = quaternion_multiply(qx, q)
        q = normalize(q)


        qy, axes = align_axis(axes, "y", global_src_up_vec)
        q = quaternion_multiply(qy, q)
        q = normalize(q)

        qx, axes = align_axis(axes, "x", global_src_x_vec)
        q = quaternion_multiply(qx, q)
        q = normalize(q)
    #else:
        #print("do not apply x", free_joint_name)
    q = normalize(q)
    return q


def find_rotation_analytically(new_skeleton, free_joint_name, target, frame, joint_cos_map, is_root=False, max_iter_count=10):
    global_src_up_vec = target["global_src_up_vec"]
    global_src_x_vec = target["global_src_x_vec"]
    local_target_axes = joint_cos_map[free_joint_name]
    if is_root:
        q = align_root_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec,global_src_x_vec, joint_cos_map, max_iter_count)
    else:
        q = align_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec, global_src_x_vec, joint_cos_map)
    return to_local_cos(new_skeleton, free_joint_name, frame, q)


def get_parent_map(joints):
    """Returns a dict of node names to their parent node's name"""
    parent_dict = dict()
    for joint_name in list(joints.keys()):
        parent_dict[joint_name] = joints[joint_name]['parent']
    return parent_dict


def get_children_map(joints):
    """Returns a dict of node names to a list of children names"""
    child_dict = dict()
    for joint_name in list(joints.keys()):
        parent_name = joints[joint_name]['parent']
        if parent_name not in child_dict:
            child_dict[parent_name] = list()
        child_dict[parent_name].append(joint_name)
    return child_dict


class PointCloudRetargeting(object):
    def __init__(self, src_joints, src_model, target_skeleton, target_to_src_joint_map, scale_factor=1.0, additional_rotation_map=None, constant_offset=None, place_on_ground=False, ground_height=0):
        self.src_joints = src_joints
        self.src_model = src_model
        self.target_skeleton = target_skeleton
        self.target_to_src_joint_map = target_to_src_joint_map
        if target_skeleton.skeleton_model["joints"]["pelvis"] is not None:
            self.target_skeleton_root = target_skeleton.skeleton_model["joints"]["pelvis"]
        else:
            self.target_skeleton_root = target_skeleton.root

        #FIXME: enable spine during retargeting
        for j in ["spine_2", "spine_1", "spine"]:
            k = self.target_skeleton.skeleton_model["joints"][j]
            self.target_to_src_joint_map[k] = None

        self.src_to_target_joint_map = {v: k for k, v in list(self.target_to_src_joint_map.items())}
        self.scale_factor = scale_factor
        self.n_params = len(self.target_skeleton.animated_joints) * 4 + 3
        self.ground_height = ground_height
        self.additional_rotation_map = additional_rotation_map
        self.src_inv_joint_map = dict((v,k) for k, v in src_model["joints"].items())
        self.src_child_map = dict()
        self.src_parent_map = get_parent_map(src_joints)
        src_children_map = get_children_map(src_joints)
        for src_name in self.src_joints:
            src_child = get_child_joint(self.src_model, self.src_inv_joint_map, src_name, src_children_map)
            if src_child is not None:
                self.src_parent_map[src_child] = src_name
                self.src_child_map[src_name] = src_child
            else:
                self.src_child_map[src_name] = None
        #print("ch",self.src_child_map)
        #for j in ["pelvis", "spine", "spine_1", "spine_2"]:
        #    if j in target_joints:
        src_joint_map = self.src_model["joints"]
        for j in ["neck", "spine_2", "spine_1", "spine"]:
            if j in src_joint_map:
                self.src_parent_map["spine_03"] = "pelvis"
        self.src_child_map[src_joint_map["pelvis"]] = src_joint_map["neck"]#"pelvis" "neck_01"

        self.constant_offset = constant_offset
        self.place_on_ground = place_on_ground
        self.temp_frame_data = dict()

        self.target_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.target_skeleton)
        if "cos_map" in target_skeleton.skeleton_model:
            self.target_cos_map.update(target_skeleton.skeleton_model["cos_map"])
        if "x_cos_fixes" in target_skeleton.skeleton_model:
            apply_manual_fixes(self.target_cos_map, target_skeleton.skeleton_model["x_cos_fixes"])

        target_joints = self.target_skeleton.skeleton_model["joints"]
        self.target_spine_joints = [target_joints[j] for j in ["neck", "spine_2", "spine_1", "spine"] if j in target_joints]#["spine_03", "neck_01"]
        self.target_ball_joints = [target_joints[j] for j in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"] if j in target_joints]# ["thigh_r", "thigh_l", "upperarm_r", "upperarm_l"]
        self.target_ankle_joints = [target_joints[j] for j in ["left_ankle", "right_ankle"] if j in target_joints]

    def estimate_src_joint_cos(self, src_name, child_name, target_name, src_frame):
        joint_idx = self.src_joints[src_name]["index"]
        child_idx = self.src_joints[child_name]["index"]
        global_src_up_vec = src_frame[child_idx] - src_frame[joint_idx]
        global_src_up_vec /= np.linalg.norm(global_src_up_vec)
        self.temp_frame_data[src_name] = global_src_up_vec
        if target_name == self.target_skeleton.skeleton_model["joints"]["pelvis"]:
            left_hip = self.src_model["joints"]["left_hip"]
            right_hip = self.src_model["joints"]["right_hip"]
            left_hip_idx = self.src_joints[left_hip]["index"]
            right_hip_idx = self.src_joints[right_hip]["index"]
            global_src_x_vec = src_frame[left_hip_idx] - src_frame[right_hip_idx]
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        elif target_name in self.target_spine_joints or target_name == "CC_Base_Waist":  # find x vector from shoulders
            left_shoulder = self.src_model["joints"]["left_shoulder"]
            right_shoulder = self.src_model["joints"]["right_shoulder"]
            left_shoulder_idx = self.src_joints[left_shoulder]["index"]
            right_shoulder_idx = self.src_joints[right_shoulder]["index"]
            global_src_x_vec = src_frame[left_shoulder_idx] - src_frame[right_shoulder_idx]
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        elif target_name in self.target_ball_joints:  # use x vector of child
            child_child_name = self.src_child_map[child_name]
            child_child_idx = self.src_joints[child_child_name]["index"]
            child_global_src_up_vec = src_frame[child_child_idx] - src_frame[child_idx]
            child_global_src_up_vec /= np.linalg.norm(child_global_src_up_vec)

            global_src_x_vec = np.cross(global_src_up_vec, child_global_src_up_vec)
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        else:  # find x vector by cross product with parent
            global_src_x_vec = None
            if src_name in self.src_parent_map:
                parent_joint = self.src_parent_map[src_name]
                if parent_joint in self.temp_frame_data:
                    global_parent_up_vector = self.temp_frame_data[parent_joint]
                    global_src_x_vec = np.cross(global_src_up_vec, global_parent_up_vector)
                    global_src_x_vec /= np.linalg.norm(global_src_x_vec)
                    # print("apply",src_name, parent_joint, global_src_x_vec)
                    # if target_name in ["calf_l", "calf_r","thigh_r","thigh_l", "spine_03","neck_01","lowerarm_r","lowerarm_l"]:
                    if target_name not in self.target_ankle_joints:
                        global_src_x_vec = - global_src_x_vec
                        # global_src_x_vec = None
                        # if global_src_x_vec is None:
                        #    print("did not find vector", target_name, parent_joint, self.target_skeleton.root)


        return {"global_src_up_vec": global_src_up_vec,
                "global_src_x_vec": global_src_x_vec}

    def rotate_bone(self, src_name, target_name, src_frame, target_frame, guess):
        q = guess
        if src_name not in self.src_child_map.keys() or self.src_child_map[src_name] is None:
            return q
        if self.src_child_map[src_name] in self.src_to_target_joint_map:#  and or target_name =="neck_01" or target_name.startswith("hand")
            child_name = self.src_child_map[src_name]
            if child_name not in self.src_joints.keys():
                return q
            src_cos = self.estimate_src_joint_cos(src_name, child_name, target_name, src_frame)
            #src_cos = self.src_cos_map[target_name]
            is_root = False
            if target_name == self.target_skeleton_root:
                is_root = True
            q = find_rotation_analytically(self.target_skeleton, target_name, src_cos, target_frame, self.target_cos_map, is_root)
        return q/np.linalg.norm(q)

    def generate_src_cos_map(self, src_frame):
        self.src_cos_map = dict()
        for target_name in self.target_skeleton.animated_joints:
            if target_name in self.target_to_src_joint_map.keys():
                src_name = self.target_to_src_joint_map[target_name]
                if src_name is not None and src_name in self.src_joints.keys():
                    if self.src_child_map[src_name] in self.src_to_target_joint_map:
                        child_name = self.src_child_map[src_name]
                        if child_name in self.src_joints.keys():
                            self.src_cos_map[target_name] = self.estimate_src_joint_cos(src_name, child_name, target_name, src_frame)

    def retarget_frame(self, src_frame, ref_frame):
        target_frame = np.zeros(self.n_params)
        self.temp_frame_data = dict()
        # copy the root translation assuming the rocketbox skeleton with static offset on the hips is used as source
        target_frame[:3] = np.array(src_frame[0]) * self.scale_factor
        if self.constant_offset is not None:
            target_frame[:3] += self.constant_offset
        animated_joints = self.target_skeleton.animated_joints
        target_offset = 3
        for target_name in animated_joints:
            q = get_quaternion_rotation_by_name(target_name, self.target_skeleton.reference_frame, self.target_skeleton, root_offset=3)
            if target_name in self.target_to_src_joint_map.keys():
                src_name = self.target_to_src_joint_map[target_name]
                if src_name is not None and src_name in self.src_joints.keys():
                    q = self.rotate_bone(src_name, target_name, src_frame, target_frame, q)
            if ref_frame is not None:
                q = q if np.dot(ref_frame[target_offset:target_offset + 4], q) >= 0 else -q
            target_frame[target_offset:target_offset + 4] = q
            target_offset += 4

        # apply offset on the root taking the orientation into account
        #aligning_root = self.target_skeleton.skeleton_model["joints"]["pelvis"]
        #target_frame = align_root_translation(self.target_skeleton, target_frame, src_frame, aligning_root, self.scale_factor)
        return target_frame

    def run(self, src_frames, frame_range):
        n_frames = len(src_frames)
        target_frames = []
        if n_frames > 0:
            #n_dims = len(src_frames[0])
            if frame_range is None:
                frame_range = (0, n_frames)

            #print("retarget", n_frames, n_dims, len(self.target_skeleton.animated_joints), frame_range)
            if self.additional_rotation_map is not None:
               src_frames = apply_additional_rotation_on_frames(self.src_skeleton.animated_joints, src_frames, self.additional_rotation_map)

            ref_frame = None
            for idx, src_frame in enumerate(src_frames[frame_range[0]:frame_range[1]]):
                target_frame = self.retarget_frame(src_frame, ref_frame)
                if ref_frame is None:
                    ref_frame = target_frame
                target_frames.append(target_frame)
            target_frames = np.array(target_frames)
            if self.place_on_ground:
                delta = target_frames[0][1] - self.ground_height
                target_frames[:, 1] -= delta
        return target_frames


def generate_joint_map(src_model, target_model):
    joint_map = dict()
    for j in src_model["joints"]:
        if j in target_model["joints"]:
            src = src_model["joints"][j]
            target = target_model["joints"][j]
            joint_map[target] = src
    return joint_map


def retarget_from_point_cloud_to_target(src_joints, src_model, target_skeleton, src_frames, joint_map=None, additional_rotation_map=None, scale_factor=1.0, frame_range=None, place_on_ground=False):
    if joint_map is None:
        joint_map = generate_joint_map(src_model, target_skeleton.skeleton_model)
    retargeting = PointCloudRetargeting(src_joints, src_model, target_skeleton, joint_map, scale_factor, additional_rotation_map=additional_rotation_map, place_on_ground=place_on_ground)
    return retargeting.run(src_frames, frame_range)
