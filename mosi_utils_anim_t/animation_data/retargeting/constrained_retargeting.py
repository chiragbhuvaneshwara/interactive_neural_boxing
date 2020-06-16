import numpy as np
from .analytical import Retargeting, generate_joint_map, apply_additional_rotation_on_frames
from ..motion_editing.hybrit_ik import HybritIK
from ...external.transformations import quaternion_from_matrix, quaternion_matrix

IK_SETTINGS = {
    "tolerance": 0.05,
    "optimization_method": "L-BFGS-B",
    "max_iterations": 1000,
    "interpolation_window": 120,
    "transition_window": 60,
    "use_euler_representation": False,
    "solving_method": "unconstrained",
    "activate_look_at": True,
    "max_retries": 5,
    "success_threshold": 5.0,
    "optimize_orientation": True,
    "elementary_action_max_iterations": 5,
    "elementary_action_optimization_eps": 1.0,
    "adapt_hands_during_carry_both": True,
    "constrain_place_orientation": False,
}
CONSTRAINED_JOINTS = ["left_wrist","right_wrist", "left_ankle", "right_ankle", "neck"]


class KeyframeConstraint(object):
    def __init__(self, frame_idx, joint_name, position, orientation=None, look_at=False):
        self.frame_idx = frame_idx
        self.joint_name = joint_name
        self.position = position
        self.orientation = orientation
        self.look_at = look_at

    def evaluate(self, skeleton, frame):
        if self.orientation is not None:
            parent_joint = skeleton.nodes[self.joint_name].parent
            if parent_joint is not None:
                m = quaternion_matrix(self.orientation)
                parent_m = parent_joint.get_global_matrix(frame, use_cache=False)
                local_m = np.dot(np.linalg.inv(parent_m), m)
                q = quaternion_from_matrix(local_m)
                idx = skeleton.animated_joints.index(parent_joint.node_name)
                # idx = skeleton.nodes[c.joint_name].quaternion_frame_index * 4
                frame[idx:idx + 4] = q
        d = self.position - skeleton.nodes[self.joint_name].get_global_position(frame)
        return np.dot(d, d)


class ConstrainedRetargeting(Retargeting):
    def __init__(self, src_skeleton, target_skeleton, target_to_src_joint_map, scale_factor=1.0, additional_rotation_map=None, constant_offset=None, place_on_ground=False, ground_height=0):
        Retargeting.__init__(self, src_skeleton, target_skeleton, target_to_src_joint_map, scale_factor, additional_rotation_map, constant_offset, place_on_ground, ground_height)
        src_joint_map = src_skeleton.skeleton_model["joints"]
        self.constrained_joints = [src_joint_map[j] for j in CONSTRAINED_JOINTS]
        self.ik = HybritIK(target_skeleton, IK_SETTINGS)
        target_joint_map = target_skeleton.skeleton_model["joints"]
        self.ik.add_analytical_ik(target_joint_map["left_wrist"], target_joint_map["left_elbow"], target_joint_map["left_shoulder"], [1,0,0],[0,0,1])
        self.ik.add_analytical_ik(target_joint_map["right_wrist"], target_joint_map["right_elbow"], target_joint_map["right_shoulder"], [1, 0, 0], [0, 0, 1])
        self.ik.add_analytical_ik(target_joint_map["right_ankle"], target_joint_map["right_knee"], target_joint_map["right_hip"], [1, 0, 0], [0, 0, 1])
        self.ik.add_analytical_ik(target_joint_map["left_ankle"], target_joint_map["left_knee"], target_joint_map["left_hip"], [1, 0, 0], [0, 0, 1])

    def generate_ik_constraints(self, frame):
        constraints = []
        for j in self.constrained_joints:
            p = self.src_skeleton.nodes[j].get_global_position(frame)
            c = KeyframeConstraint(0,j, p)
            constraints.append(c)
        return constraints

    def run(self, src_frames, frame_range):
        n_frames = len(src_frames)
        target_frames = []
        if n_frames > 0:
            if frame_range is None:
                frame_range = (0, n_frames)

            if self.additional_rotation_map is not None:
               src_frames = apply_additional_rotation_on_frames(self.src_skeleton.animated_joints, src_frames, self.additional_rotation_map)

            ref_frame = None
            for idx, src_frame in enumerate(src_frames[frame_range[0]:frame_range[1]]):
                print("retarget frame", idx)
                ik_constraints = self.generate_ik_constraints(src_frame)
                target_frame = self.retarget_frame(src_frame, ref_frame)
                target_frame = self.ik.modify_frame(target_frame, ik_constraints)
                if ref_frame is None:
                    ref_frame = target_frame
                target_frames.append(target_frame)
            target_frames = np.array(target_frames)
            if self.place_on_ground:
                delta = target_frames[0][1] - self.ground_height
                target_frames[:,1] -= delta
        return target_frames


def retarget_from_src_to_target(src_skeleton, target_skeleton, src_frames, joint_map=None,
                                additional_rotation_map=None, scale_factor=1.0, frame_range=None,
                                place_on_ground=False):
    if joint_map is None:
        joint_map = generate_joint_map(src_skeleton.skeleton_model, target_skeleton.skeleton_model)
    retargeting = ConstrainedRetargeting(src_skeleton, target_skeleton, joint_map, scale_factor,
                              additional_rotation_map=additional_rotation_map, place_on_ground=place_on_ground)
    return retargeting.run(src_frames, frame_range)
