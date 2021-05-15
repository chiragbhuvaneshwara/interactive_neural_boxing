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


def create_local_cos_map_using_child_map(skeleton, up_vector, x_vector, child_map=None):
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = up_vector
        joint_cos_map[j]["x"] = x_vector

        if j == skeleton.root:
            joint_cos_map[j]["x"] = (-np.array(x_vector)).tolist()
        else:
            o = np.array([0, 0, 0])
            if child_map is not None and j in child_map:
                child_name = child_map[j]
                node = skeleton.nodes[child_name]
                o = np.array(node.offset)
            elif len(skeleton.nodes[j].children) > 0:
                node = skeleton.nodes[j].children[0]
                o = np.array(node.offset)
            o = normalize(o)
            if sum(o * o) > 0:
                joint_cos_map[j]["y"] = o
    return joint_cos_map


def create_local_cos_map(skeleton, up_vector, x_vector):
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = up_vector
        joint_cos_map[j]["x"] = x_vector
        if j == skeleton.root:
            joint_cos_map[j]["x"] = (-np.array(x_vector)).tolist()
    return joint_cos_map


def get_body_x_axis(skeleton):
    rh = skeleton.skeleton_model["joints"]["right_hip"]
    lh = skeleton.skeleton_model["joints"]["left_hip"]
    return get_body_axis(skeleton, rh, lh)

def get_body_y_axis(skeleton):
    a = skeleton.skeleton_model["joints"]["pelvis"]
    b = skeleton.skeleton_model["joints"]["head"]
    return get_body_axis(skeleton, a,b)

def get_quaternion_to_axis(skeleton, joint_a, joint_b, axis):
    ident_f = skeleton.identity_frame
    ap = skeleton.nodes[joint_a].get_global_position(ident_f)
    bp = skeleton.nodes[joint_b].get_global_position(ident_f)
    delta = bp - ap
    delta /= np.linalg.norm(delta)
    return quaternion_from_vector_to_vector(axis, delta)


def get_body_axis(skeleton, joint_a, joint_b, project=True):
    ident_f = skeleton.identity_frame
    ap = skeleton.nodes[joint_a].get_global_position(ident_f)
    bp = skeleton.nodes[joint_b].get_global_position(ident_f)
    delta = bp - ap
    m = np.linalg.norm(delta)
    if m != 0:
        delta /= m
        if project:
            projection = project_vector_on_axis(delta)
            return projection / np.linalg.norm(projection)
        else:
            return delta
    else:
        return None

def create_local_cos_map_from_skeleton(skeleton):
    body_x_axis = get_body_x_axis(skeleton)
    print("body x axis", body_x_axis)
    body_y_axis = get_body_y_axis(skeleton)
    print("body y axis", body_y_axis)
    joints = skeleton.skeleton_model["joints"]
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        node = skeleton.nodes[j]
        if np.linalg.norm(node.offset) > 0:
            y_axis = node.offset / np.linalg.norm(node.offset)
            #projected_offset = project_vector_on_axis(node.offset)
            #y_axis = projected_offset/np.linalg.norm(projected_offset)
            #if j in [joints["left_hip"], joints["right_hip"],joints["pelvis"]]:
            #    y_axis = get_body_axis(skeleton, j, node.children[0].node_name)
                #print(j, node.children[0].node_name, y_axis)
            joint_cos_map[j]["y"] = y_axis
        else:
            joint_cos_map[j]["y"] = body_y_axis
        joint_cos_map[j]["x"] = body_x_axis
        z_vector = np.cross(joint_cos_map[j]["y"], joint_cos_map[j]["x"])
        if np.linalg.norm(z_vector) == 0.0:
            joint_cos_map[j]["x"] = body_y_axis * -np.sum(joint_cos_map[j]["y"])

    return joint_cos_map


def create_local_cos_map_from_skeleton_axes_old(skeleton, flip=1.0):
    body_x_axis = get_body_x_axis(skeleton)*flip
    print("body x axis", body_x_axis)
    body_y_axis = get_body_y_axis(skeleton)
    print("body y axis", body_y_axis)
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = body_y_axis
        joint_cos_map[j]["x"] = body_x_axis
        node = skeleton.nodes[j]
        if len(node.children) > 0 and np.linalg.norm(node.children[0].offset) > 0 and j != skeleton.root:
            y_axis = get_body_axis(skeleton, j, node.children[0].node_name)
            joint_cos_map[j]["y"] = y_axis
            #check if the new y axis is similar to the x axis
            z_vector = np.cross(y_axis, joint_cos_map[j]["x"])
            if np.linalg.norm(z_vector) == 0.0:
                joint_cos_map[j]["x"] = body_y_axis *-np.sum(joint_cos_map[j]["y"])
            #check for angle and rotate
            q = get_quaternion_to_axis(skeleton, j, node.children[0].node_name, y_axis)
            m = quaternion_matrix(q)[:3, :3]
            for key, a in list(joint_cos_map[j].items()):
                joint_cos_map[j][key] = np.dot(m, a)
                joint_cos_map[j][key] = normalize(joint_cos_map[j][key])
            #print(j, joint_cos_map[j])
    return joint_cos_map


def rotate_axes(cos, q):
    m = quaternion_matrix(q)[:3, :3]
    for key, a in list(cos.items()):
        cos[key] = np.dot(m, a)
        cos[key] = normalize(cos[key])
    return cos

def create_local_cos_map_from_skeleton_axes(skeleton, flip=1.0, project=True):#TO DO fix bug
    body_x_axis = get_body_x_axis(skeleton)*flip
    print("body x axis", body_x_axis)
    body_y_axis = get_body_y_axis(skeleton)
    print("body y axis", body_y_axis)
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = body_y_axis
        joint_cos_map[j]["x"] = body_x_axis
        node = skeleton.nodes[j]
        n_children = len(node.children)
        child_idx = -1# for retargeting von iclone auf game engine?
        if n_children > 0 and j != skeleton.root:
            #pick the child index based on heuristic for game engine skeleton
            if n_children ==3 and j == "spine_03":
                child_idx = 2
            if np.linalg.norm(node.children[child_idx].offset) == 0:
                continue
            y_axis = get_body_axis(skeleton, j, node.children[child_idx].node_name, project)
            joint_cos_map[j]["y"] = y_axis
            #check if the new y axis is similar to the x axis
            z_vector = np.cross(y_axis, joint_cos_map[j]["x"])
            if np.linalg.norm(z_vector) == 0.0:
                joint_cos_map[j]["x"] = body_y_axis *-np.sum(joint_cos_map[j]["y"])
            #check for angle and rotate
            q = get_quaternion_to_axis(skeleton, j, node.children[child_idx].node_name, y_axis)
            rotate_axes(joint_cos_map[j], q)
        #print(j, joint_cos_map[j])
    return joint_cos_map

def get_child_joint(skeleton, inv_joint_map, node_name):
    """ Warning output is random if there are more than one child joints
        and the value is not specified in the JOINT_CHILD_MAP """
    child_node = None
    if len(skeleton.nodes[node_name].children) > 0:
        child_node = skeleton.nodes[node_name].children[-1]
    if node_name in inv_joint_map:
        joint_name = inv_joint_map[node_name]
        while joint_name in JOINT_CHILD_MAP:
            child_joint_name = JOINT_CHILD_MAP[joint_name]

            # check if child joint is mapped
            joint_key = None
            if child_joint_name in skeleton.skeleton_model["joints"]:
                joint_key = skeleton.skeleton_model["joints"][child_joint_name]

            if joint_key is not None: # return child joint
                child_node = skeleton.nodes[joint_key]
                return child_node
            else: #keep traversing until end of child map is reached
                if child_joint_name in JOINT_CHILD_MAP:
                    joint_name = JOINT_CHILD_MAP[child_joint_name]
                    #print(joint_name)
                else:
                    break
    return child_node

def create_local_cos_map_from_skeleton_axes_with_map(skeleton, flip=1.0, project=True):
    body_x_axis = get_body_x_axis(skeleton)*flip
    #print("body x axis", body_x_axis)
    body_y_axis = get_body_y_axis(skeleton)
    #print("body y axis", body_y_axis)
    inv_joint_map = dict((v,k) for k, v in skeleton.skeleton_model["joints"].items())
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = body_y_axis
        joint_cos_map[j]["x"] = body_x_axis

        node = skeleton.nodes[j]
        child_node = get_child_joint(skeleton, inv_joint_map, node.node_name)
        if child_node is None:
            continue

        y_axis = get_body_axis(skeleton, j, child_node.node_name, project)
        if y_axis is not None:
            joint_cos_map[j]["y"] = y_axis
            #check if the new y axis is similar to the x axis
            z_vector = np.cross(y_axis, joint_cos_map[j]["x"])
            if np.linalg.norm(z_vector) == 0.0:
                joint_cos_map[j]["x"] = body_y_axis * -np.sum(joint_cos_map[j]["y"])
            #check for angle and rotate
            q = get_quaternion_to_axis(skeleton, j, child_node.node_name, y_axis)
            rotate_axes(joint_cos_map[j], q)
        else:
            joint_cos_map[j]["y"] = None
            joint_cos_map[j]["x"] = None
    return joint_cos_map


def create_local_cos_map_from_skeleton_rocketbox(skeleton):
    joint_cos_map = create_local_cos_map_from_skeleton_axes(skeleton, -1.0)
    joint_cos_map["Hips"]["x"] = [0, -1, 0]
    joint_cos_map["Spine"]["x"] = [0, -1, 0]
    joint_cos_map["Spine_1"]["x"] = [0, -1, 0]
    joint_cos_map["Neck"]["x"] = [0, -1, 0]
    return joint_cos_map

def create_local_cos_map_from_skeleton_mcs(skeleton):
    joint_cos_map = create_local_cos_map_from_skeleton_axes(skeleton)
    joint_cos_map["LeftShoulder"]["x"] = [-1, 0, 0]
    joint_cos_map["RightShoulder"]["x"] = [-1, 0, 0]
    joint_cos_map["LeftElbow"]["x"] = [-1, 0, 0]
    joint_cos_map["RightElbow"]["x"] = [-1, 0, 0]
    return joint_cos_map

def create_local_cos_map_from_skeleton_iclone(skeleton):
    joint_cos_map = create_local_cos_map_from_skeleton_axes_with_map(skeleton)
    joint_cos_map["CC_Base_L_Thigh"]["x"] *= -1
    joint_cos_map["CC_Base_R_Thigh"]["x"] *= -1
    joint_cos_map["CC_Base_L_Calf"]["x"] *= -1
    joint_cos_map["CC_Base_R_Calf"]["x"] *= -1
    joint_cos_map["CC_Base_L_Foot"]["x"] *= -1
    joint_cos_map["CC_Base_R_Foot"]["x"] *= -1
    return joint_cos_map


X_JOINTS = ["CC_Base_L_Thigh", "CC_Base_R_Thigh", "CC_Base_L_Calf", "CC_Base_R_Calf", "CC_Base_L_Foot", "CC_Base_R_Foot",
           "Hips", "Spine", "Spine_1", "Neck"]
def apply_manual_fixes(joint_cos_map, joints=X_JOINTS):
    for j in joints:
        if j in joint_cos_map:
            joint_cos_map[j]["x"] *= -1



def align_root_joint(axes, global_src_x_vec, max_iter_count=10):
    # handle special case for the root joint
    # apply only the y axis rotation of the Hip to the Game_engine node
    not_aligned = True
    q = [1, 0, 0, 0]
    iter_count = 0
    while not_aligned:
        qx, axes = align_axis(axes, "x", global_src_x_vec)  # first find rotation to align x axis
        q = quaternion_multiply(qx, q)
        q = normalize(q)
        qy, axes = align_axis(axes, "y", OPENGL_UP_AXIS)  # then add a rotation to let the y axis point up
        q = quaternion_multiply(qy, q)
        q = normalize(q)

        a_y = math.acos(np.dot(axes["y"], OPENGL_UP_AXIS))
        a_x = math.acos(np.dot(axes["x"], global_src_x_vec))
        iter_count += 1
        not_aligned = a_y > 0.1 or a_x > 0.1 and iter_count < max_iter_count
    return q


def align_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec, global_src_x_vec, joint_cos_map, apply_spine_fix=False):
    # first align the bone vectors
    q = [1, 0, 0, 0]
    qy, axes = align_axis(local_target_axes, "y", global_src_up_vec)
    q = quaternion_multiply(qy, q)
    joint_map = new_skeleton.skeleton_model["joints"]
    q = normalize(q)
    spine_joints = [joint_map["pelvis"]]#, joint_map["spine"], joint_map["spine_1"]
    if "spine" in joint_map:
        spine_joints += joint_map["spine"]
    if "spine_1" in joint_map:
        spine_joints += joint_map["spine_1"]
    if "spine_2" in joint_map:
        spine_joints += joint_map["spine_2"]
    if free_joint_name in spine_joints and apply_spine_fix :#FIX ME breaks the cma es trajectory following for some skeletons if set to True
        # handle special case of applying the x axis rotation of the Hip to the pelvis
        node = new_skeleton.nodes[free_joint_name]
        t_pose_global_m = node.get_global_matrix(new_skeleton.reference_frame)
        global_original = np.dot(t_pose_global_m[:3, :3], joint_cos_map[free_joint_name]["y"])
        global_original = normalize(global_original)

        neck_node_name = new_skeleton.skeleton_model["joints"]["neck"]
        neck_pos = new_skeleton.nodes[neck_node_name].get_global_position(new_skeleton.reference_frame)
        pelvis_pos = t_pose_global_m[:3, 3]
        direction_to_neck = neck_pos-pelvis_pos
        direction_to_neck /= np.linalg.norm(direction_to_neck)
        qoffset = find_rotation_between_vectors(direction_to_neck, global_original)
        q = quaternion_multiply(qoffset, q)
        q = normalize(q)

    # then align the twisting angles
    qx, axes = align_axis(axes, "x", global_src_x_vec)
    q = quaternion_multiply(qx, q)
    q = normalize(q)
    return q


def find_rotation_analytically(new_skeleton, free_joint_name, target, frame, joint_cos_map, apply_spine_fix=False, max_iter_count=10):
    global_src_up_vec = target["global_src_up_vec"]
    global_src_x_vec = target["global_src_x_vec"]

    local_target_axes = joint_cos_map[free_joint_name]
    #FIX ME captury to custom for hybrit needs the align_root method
    #if free_joint_name == new_skeleton.root:
    #    q = align_root_joint(local_target_axes,  global_src_x_vec, max_iter_count)
    #else:
    q = align_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec, global_src_x_vec, joint_cos_map, apply_spine_fix=apply_spine_fix)
    return to_local_cos(new_skeleton, free_joint_name, frame, q)


class Retargeting(object):
    def __init__(self, src_skeleton, target_skeleton, target_to_src_joint_map, scale_factor=1.0, additional_rotation_map=None, constant_offset=None, place_on_ground=False, ground_height=0):
        self.src_skeleton = src_skeleton
        self.target_skeleton = target_skeleton
        self.target_to_src_joint_map = target_to_src_joint_map
        self.src_to_target_joint_map = {v: k for k, v in list(self.target_to_src_joint_map.items())}
        self.scale_factor = scale_factor
        self.n_params = len(self.target_skeleton.animated_joints) * 4 + 3
        self.ground_height = ground_height
        self.additional_rotation_map = additional_rotation_map
        self.src_inv_joint_map = dict((v,k) for k, v in src_skeleton.skeleton_model["joints"].items())
        self.src_child_map = dict()
        for src_name in self.src_skeleton.animated_joints:
            src_child = get_child_joint(self.src_skeleton, self.src_inv_joint_map, src_name)
            if src_child is not None:
                self.src_child_map[src_name] = src_child.node_name
            else:
                self.src_child_map[src_name] = None
        self.target_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.target_skeleton)
        self.src_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.src_skeleton, flip=1.0, project=True)

        if "cos_map" in target_skeleton.skeleton_model:
            self.target_cos_map.update(target_skeleton.skeleton_model["cos_map"])
        if "x_cos_fixes" in target_skeleton.skeleton_model:
            apply_manual_fixes(self.target_cos_map, target_skeleton.skeleton_model["x_cos_fixes"])
        if "cos_map" in src_skeleton.skeleton_model:
            self.src_cos_map.update(src_skeleton.skeleton_model["cos_map"])
        if "x_cos_fixes" in src_skeleton.skeleton_model:
            apply_manual_fixes(self.src_cos_map, src_skeleton.skeleton_model["x_cos_fixes"])
        self.correction_map = dict()
        self.create_correction_map()
        self.constant_offset = constant_offset
        self.place_on_ground = place_on_ground
        self.apply_spine_fix = self.src_skeleton.animated_joints != self.target_skeleton.animated_joints

    def create_correction_map(self):
        self.correction_map = dict()
        joint_map = self.target_skeleton.skeleton_model["joints"]
        for target_name in self.target_to_src_joint_map:
            src_name = self.target_to_src_joint_map[target_name]
            if src_name in self.src_cos_map and target_name is not None:
                src_zero_vector_y = self.src_cos_map[src_name]["y"]
                target_zero_vector_y = self.target_cos_map[target_name]["y"]
                src_zero_vector_x = self.src_cos_map[src_name]["x"]
                target_zero_vector_x = self.target_cos_map[target_name]["x"]
                if target_zero_vector_y is not None and src_zero_vector_y is not None:
                    q = quaternion_from_vector_to_vector(target_zero_vector_y, src_zero_vector_y)
                    q = normalize(q)

                    if target_name in [joint_map["pelvis"], joint_map["spine"], joint_map["spine_1"]]:#,, joint_map["spine_2"]
                        # add offset rotation to spine based on an upright reference pose
                        m = quaternion_matrix(q)[:3, :3]
                        v = normalize(np.dot(m, target_zero_vector_y))
                        node = self.target_skeleton.nodes[target_name]
                        t_pose_global_m = node.get_global_matrix(self.target_skeleton.reference_frame)[:3, :3]
                        global_original = np.dot(t_pose_global_m, v)
                        global_original = normalize(global_original)
                        qoffset = find_rotation_between_vectors(OPENGL_UP_AXIS, global_original)
                        q = quaternion_multiply(qoffset, q)
                        q = normalize(q)

                    m = quaternion_matrix(q)[:3, :3]
                    target_zero_vector_x = normalize(np.dot(m, target_zero_vector_x))
                    qx = quaternion_from_vector_to_vector(target_zero_vector_x, src_zero_vector_x)
                    q = quaternion_multiply(qx, q)
                    q = normalize(q)
                    self.correction_map[target_name] = q

    def rotate_bone_old(self, src_name, target_name, src_frame, target_frame, guess):
        q = guess
        if src_name not in self.src_child_map.keys() or self.src_child_map[src_name] is None:
            #print("dont map1", src_name, target_name)
            return q
        src_x_axis = self.src_cos_map[src_name]["x"]
        src_up_axis = self.src_cos_map[src_name]["y"]
        if self.src_child_map[src_name] in self.src_to_target_joint_map and self.src_cos_map[src_name]["y"] is not None and self.target_cos_map[target_name]["y"] is not None:#  and or target_name =="neck_01" or target_name.startswith("hand")
            #print("map", src_name, target_name, src_up_axis,  self.target_cos_map[target_name]["y"], self.src_child_map[src_name])
            global_m = self.src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
            global_src_up_vec = normalize(np.dot(global_m, src_up_axis))
            global_src_x_vec = normalize(np.dot(global_m, src_x_axis))

            target = {"global_src_up_vec": global_src_up_vec,
                      "global_src_x_vec": global_src_x_vec}
            q = find_rotation_analytically(self.target_skeleton, target_name, target, target_frame, self.target_cos_map, self.apply_spine_fix)
        #else:
        #    print("dont map2", src_name, target_name, self.src_child_map[src_name], self.src_child_map[src_name] in self.src_to_target_joint_map)
        return q

    def rotate_bone(self, src_name, target_name, src_frame, target_frame, quess):
        q = quess
        src_child_name = self.src_skeleton.nodes[src_name].children[0].node_name
        if src_child_name in self.src_to_target_joint_map:
            m = self.src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
            gq = quaternion_from_matrix(m)
            correction_q = self.correction_map[target_name]
            q = quaternion_multiply(gq, correction_q)
            q = normalize(q)
            q = to_local_cos(self.target_skeleton, target_name, target_frame, q)
        return q

    def retarget_frame(self, src_frame, ref_frame):
        target_frame = np.zeros(self.n_params)
        # copy the root translation assuming the rocketbox skeleton with static offset on the hips is used as source
        target_frame[0] = src_frame[0] * self.scale_factor
        target_frame[1] = src_frame[1] * self.scale_factor
        target_frame[2] = src_frame[2] * self.scale_factor

        if self.constant_offset is not None:
            target_frame[:3] += self.constant_offset
        animated_joints = self.target_skeleton.animated_joints
        target_offset = 3

        for target_name in animated_joints:
            q = get_quaternion_rotation_by_name(target_name, self.target_skeleton.reference_frame, self.target_skeleton, root_offset=3)

            if target_name in self.target_to_src_joint_map.keys():

                src_name = self.target_to_src_joint_map[target_name]
                if src_name is not None and len(self.src_skeleton.nodes[src_name].children)>0:
                    #q = self.rotate_bone(src_name, target_name, src_frame, target_frame, q)
                    q = self.rotate_bone_old(src_name, target_name, src_frame, target_frame, q)


            if ref_frame is not None:
                #  align quaternion to the reference frame to allow interpolation
                #  http://physicsforgames.blogspot.de/2010/02/quaternions.html
                ref_q = ref_frame[target_offset:target_offset + 4]
                if np.dot(ref_q, q) < 0:
                    q = -q
            target_frame[target_offset:target_offset + 4] = q
            target_offset += 4

        # apply offset on the root taking the orientation into account
        aligning_root = self.target_skeleton.skeleton_model["joints"]["pelvis"]
        target_frame = align_root_translation(self.target_skeleton, target_frame, src_frame, aligning_root, self.scale_factor)
        return target_frame

    def run(self, src_frames, frame_range):
        #TO DO get up axes and cross vector from skeleton heuristically,
        # bone_dir = up, left leg to right leg = cross for all bones
        n_frames = len(src_frames)
        target_frames = []
        if n_frames > 0:
            if frame_range is None:
                frame_range = (0, n_frames)
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
                target_frames[:,1] -= delta
        return target_frames


def generate_joint_map(src_model, target_model, joint_filter=None):
    print(target_model.keys())
    joint_map = dict()
    for j in src_model["joints"]:
        if joint_filter is not None and j not in joint_filter:
            continue
        if j in target_model["joints"]:
            src = src_model["joints"][j]
            target = target_model["joints"][j]
            joint_map[target] = src
    return joint_map


def retarget_from_src_to_target(src_skeleton, target_skeleton, src_frames, joint_map=None, additional_rotation_map=None, scale_factor=1.0, frame_range=None, place_on_ground=False, joint_filter=None):
    if joint_map is None:
        joint_map = generate_joint_map(src_skeleton.skeleton_model, target_skeleton.skeleton_model, joint_filter)
    retargeting = Retargeting(src_skeleton, target_skeleton, joint_map, scale_factor, additional_rotation_map=additional_rotation_map, place_on_ground=place_on_ground)
    return retargeting.run(src_frames, frame_range)
