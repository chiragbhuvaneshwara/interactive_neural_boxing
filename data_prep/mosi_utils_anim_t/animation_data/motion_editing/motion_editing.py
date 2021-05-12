from copy import copy
import numpy as np
import collections
from .numerical_ik_quat import NumericalInverseKinematicsQuat
from .numerical_ik_exp import NumericalInverseKinematicsExp
from .cubic_motion_spline import CubicMotionSpline
from ..motion_blending import smooth_joints_around_transition_using_slerp, create_transition_using_slerp
from ...external.transformations import euler_from_matrix, quaternion_multiply
from ...utilities.log import write_message_to_log, LOG_MODE_DEBUG
from .utils import convert_exp_frame_to_quat_frame
from ..joint_constraints import JointConstraint, HingeConstraint2, BallSocketConstraint, ConeConstraint, ShoulderConstraint
from ...external.transformations import quaternion_matrix, quaternion_from_matrix
from .coordinate_cyclic_descent import LOOK_AT_DIR, SPINE_LOOK_AT_DIR
from ..motion_blending import smooth_quaternion_frames



def add_frames(skeleton, a, b):
    """ returns c = a + b"""
    #print("add frames", len(a), len(b))
    c = np.zeros(len(a))
    c[:3] = a[:3] + b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx * 4 + 3
        q_a = a[o:o + 4]
        q_b = b[o:o + 4]
        q_prod = quaternion_multiply(q_a, q_b)
        c[o:o + 4] = q_prod / np.linalg.norm(q_prod)
    return c


class KeyframeConstraint(object):
    def __init__(self, frame_idx, joint_name, position, orientation=None, look_at=False, offset=None, look_at_pos=None):
        self.frame_idx = frame_idx
        self.joint_name = joint_name
        self.position = position
        self.orientation = orientation
        self.look_at = look_at
        self.look_at_pos = look_at_pos
        self.offset = offset
        self.inside_region = False
        self.end_of_region = False
        self.inside_region_orientation = False
        self.keep_orientation = False

        # set in case it is a relative constraint
        self.relative_parent_joint_name = None # joint the offsets points from to the target
        self.relative_offset = None

    def instantiate_relative_constraint(self, skeleton, frame):
        """ turn relative constraint into a normal constraint"""
        ppos = skeleton.nodes[self.relative_parent_joint_name].get_global_position(frame)
        pos = ppos #+ self.relative_offset
        return KeyframeConstraint(self.frame_idx, self.joint_name, pos)

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
        if self.offset is not None:
            m = skeleton.nodes[self.joint_name].get_global_matrix(frame)
            p = np.dot(m, self.offset)[:3]
            d = self.position - p
        else:
            d = self.position - skeleton.nodes[self.joint_name].get_global_position(frame)
        return np.dot(d, d)


class MotionEditing(object):
    def __init__(self, skeleton, algorithm_settings):
        self.skeleton = skeleton
        self._ik_settings = algorithm_settings["inverse_kinematics_settings"]
        self.window = int(self._ik_settings["interpolation_window"])
        self.transition_window = int(self._ik_settings["transition_window"])
        self.verbose = False
        self.use_euler = self._ik_settings["use_euler_representation"]
        self.solving_method = self._ik_settings["solving_method"]
        self.success_threshold = self._ik_settings["success_threshold"]
        self.max_retries = int(self._ik_settings["max_retries"])
        self.activate_look_at = self._ik_settings["activate_look_at"]
        self.optimize_orientation = self._ik_settings["optimize_orientation"]
        self.elementary_action_max_iterations = int(self._ik_settings["elementary_action_max_iterations"])
        self.elementary_action_epsilon = self._ik_settings["elementary_action_optimization_eps"]
        self.adapt_hands_during_both_hand_carry = self._ik_settings["adapt_hands_during_carry_both"]
        self.pose = SkeletonPoseModel(self.skeleton, self.use_euler)
        self._ik = NumericalInverseKinematicsQuat(self.pose, self._ik_settings)
        self._ik_exp = NumericalInverseKinematicsExp(self.skeleton, self._ik_settings)

    def add_constraints_to_skeleton(self, joint_constraints):
        joint_map = self.skeleton.skeleton_model["joints"]
        for j in joint_constraints:
            if j in joint_map:
                skel_j = joint_map[j]
            else:
                continue
            c = joint_constraints[j]
            if c["type"] == "static":
                h = JointConstraint()
                h.is_static = True
                self.skeleton.nodes[skel_j].joint_constraint = h
            elif c["type"] == "hinge":
                swing_axis = np.array(c["swing_axis"])
                twist_axis = np.array(c["twist_axis"])
                deg_angle_range = None
                if "k1" in c and "k2" in c:
                    deg_angle_range = [c["k1"], c["k2"]]
                print("add hinge constraint to", skel_j)
                h = HingeConstraint2(swing_axis, twist_axis, deg_angle_range)
                self.skeleton.nodes[skel_j].joint_constraint = h
            elif c["type"] == "ball":
                axis = np.array(c["axis"])
                k = c["k"]
                print("add ball socket constraint to", skel_j)
                h = BallSocketConstraint(axis, k)
                self.skeleton.nodes[skel_j].joint_constraint = h
            elif c["type"] == "cone":
                axis = np.array(c["axis"])
                k = c["k"]
                print("add cone constraint to", skel_j)
                h = ConeConstraint(axis, k)
                self.skeleton.nodes[skel_j].joint_constraint = h
            elif c["type"] == "shoulder":
                axis = np.array(c["axis"])
                k = c["k"]
                k1 = c["k1"]
                k2 = c["k2"]
                print("add shoulder socket constraint to", skel_j)
                h = ShoulderConstraint(axis, k, k1, k2)
                self.skeleton.nodes[skel_j].joint_constraint = h

    def modify_motion_vector(self, motion_vector):
        for idx, action_ik_constraints in enumerate(motion_vector.ik_constraints):
            write_message_to_log("Apply IK to elementary action " + str(idx), LOG_MODE_DEBUG)
            self._optimize_action_ik_constraints(motion_vector, action_ik_constraints)

    def _optimize_action_ik_constraints(self, motion_vector, action_ik_constraints):
        i = 0
        last_error = None
        keep_running = True
        trajectory_weights = 1.0
        # modify individual keyframes based on constraints
        while keep_running:
            error = 0.0
            if "trajectories" in list(action_ik_constraints.keys()):
                constraints = action_ik_constraints["trajectories"]
                c_error = self._modify_motion_vector_using_trajectory_constraint_list(motion_vector, constraints)
                error += c_error * trajectory_weights
            if "keyframes" in list(action_ik_constraints.keys()):
                constraints = action_ik_constraints["keyframes"]
                error += self._modify_motion_vector_using_keyframe_constraint_list(motion_vector, constraints)
            if last_error is not None:
                delta = abs(last_error - error)
            else:
                delta = np.inf
            last_error = error
            i += 1
            keep_running = i < self.elementary_action_max_iterations and delta > self.elementary_action_epsilon
            write_message_to_log("IK iteration " + str(i) + " " + str(error) + " " + str(delta) + " " + str(
                self.elementary_action_epsilon), LOG_MODE_DEBUG)

    def _modify_motion_vector_using_keyframe_constraint_list(self, motion_vector, constraints):
        error = 0.0
        for keyframe, keyframe_constraints in list(constraints.items()):
            keyframe = int(keyframe)
            if "single" in list(keyframe_constraints.keys()):
                for c in keyframe_constraints["single"]:
                    if c.optimize:
                        if c.frame_range is not None:
                            error += self._modify_motion_vector_using_keyframe_constraint_range(motion_vector, c,
                                                                                                c.frame_range)
                        else:
                            error += self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)
                    start = keyframe
                    end = keyframe + 1
                    if self.activate_look_at and c.look_at:
                        self._look_at_in_range(motion_vector.frames, c.position, start, end)
                    print("set hand orientation", c.orientation)
                    if c.orientation is not None and self.optimize_orientation:
                        self._set_hand_orientation(motion_vector.frames, c.orientation, c.joint_name, keyframe, start, end)
        return error

    def _modify_frame_using_keyframe_constraint(self, motion_vector, constraint, keyframe):
        self.set_pose_from_frame(motion_vector.frames[keyframe])
        error = self._ik.modify_pose_general(constraint)
        motion_vector.frames[keyframe] = self.pose.get_vector()
        if self.window > 0:
            self.interpolate_around_keyframe(motion_vector.frames, constraint.get_joint_names(), keyframe, self.window)
        return error

    def _modify_motion_vector_using_keyframe_constraint_range(self, motion_vector, constraint, frame_range):
        error = 0.0
        for frame in range(frame_range[0], frame_range[1] + 1):
            self.set_pose_from_frame(motion_vector.frames[frame])
            error += self._ik.modify_pose_general(constraint)
            motion_vector.frames[frame] = self.pose.get_vector()

        self._create_transition_for_frame_range(motion_vector.frames, frame_range[0], frame_range[1],
                                                self.pose.free_joints_map[constraint.joint_name])
        return error

    def interpolate_around_keyframe(self, frames, joint_names, keyframe, window):
        write_message_to_log("Smooth and interpolate" + str(joint_names), LOG_MODE_DEBUG)
        for target_joint_name in joint_names:
            joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[target_joint_name])
            for joint_name in self.pose.free_joints_map[target_joint_name]:
                smooth_joints_around_transition_using_slerp(frames, joint_parameter_indices[joint_name], keyframe, window)

    def _look_at_in_range(self, frames, position, start, end):
        start = max(0, start)
        end = min(frames.shape[0], end)
        for idx in range(start, end):
            self.set_pose_from_frame(frames[idx])
            self.pose.lookat(position)
            frames[idx] = self.pose.get_vector()
        self._create_transition_for_frame_range(frames, start, end - 1, [self.pose.head_joint])

    def _create_transition_for_frame_range(self, frames, start, end, target_joints):
        for target_joint in target_joints:
            joint_parameter_indices = list(range(*self.pose.extract_parameters_indices(target_joint)))
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]) - 1
            create_transition_using_slerp(frames, transition_start, start, joint_parameter_indices)
            create_transition_using_slerp(frames, end, transition_end, joint_parameter_indices)

    def _set_hand_orientation(self, frames, orientation, joint_name, keyframe, start, end):
        parent_joint_name = self.pose.get_parent_joint(joint_name)
        self.set_pose_from_frame(frames[keyframe])
        self.pose.set_hand_orientation(parent_joint_name, orientation)
        start = max(0, start)
        end = min(frames.shape[0], end)
        self._create_transition_for_frame_range(frames, start, end - 1, [parent_joint_name])

    def set_pose_from_frame(self, reference_frame):
        self.pose.set_pose_parameters(reference_frame)
        self.pose.clear_cache()

    def _extract_free_parameter_indices(self, free_joints):
        """get parameter indices of joints from reference frame
        """
        indices = {}
        for joint_name in free_joints:
            indices[joint_name] = list(range(*self.pose.extract_parameters_indices(joint_name)))
        return indices

    def _modify_motion_vector_using_trajectory_constraint_list(self, motion_vector, constraints):
        error = 0.0
        for c in constraints:
            if c["fixed_range"]:
                error += self._modify_motion_vector_using_trajectory_constraint(motion_vector, c)
            else:
                error += self._modify_motion_vector_using_trajectory_constraint_search_start(motion_vector, c)
        return error

    def _modify_motion_vector_using_trajectory_constraint(self, motion_vector, traj_constraint):
        error_sum = 0.0
        d = traj_constraint["delta"]
        trajectory = traj_constraint["trajectory"]
        start_idx = traj_constraint["start_frame"]
        end_idx = traj_constraint["end_frame"] - 1
        end_idx = min(len(motion_vector.frames) - 1, end_idx)
        n_frames = end_idx - start_idx + 1
        target_direction = None
        if traj_constraint["constrain_orientation"]:
            target_direction = trajectory.get_direction()
            if np.linalg.norm(target_direction) == 0:
                target_direction = None

        full_length = n_frames * d
        for idx in range(n_frames):
            t = (idx * d) / full_length
            target_position = trajectory.query_point_by_parameter(t)
            keyframe = start_idx + idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._ik.modify_pose(traj_constraint["joint_name"], target_position, target_direction)
                iter_counter += 1
            error_sum += error
            motion_vector.frames[keyframe] = self.pose.get_vector()
        parent_joint = self.pose.get_parent_joint(traj_constraint["joint_name"])

        if traj_constraint["joint_name"] in list(self.pose.free_joints_map.keys()):
            free_joints = self.pose.free_joints_map[traj_constraint["joint_name"]]
            free_joints = list(set(free_joints + [parent_joint]))
        else:
            free_joints = [parent_joint]
        self._create_transition_for_frame_range(motion_vector.frames, start_idx, end_idx, free_joints)
        return error_sum

    def _modify_motion_vector_using_trajectory_constraint_search_start(self, motion_vector, traj_constraint):
        error_sum = 0.0
        trajectory = traj_constraint["trajectory"]
        start_target = trajectory.query_point_by_parameter(0.0)
        start_idx = self._find_corresponding_frame(motion_vector,
                                                   traj_constraint["start_frame"],
                                                   traj_constraint["end_frame"],
                                                   traj_constraint["joint_name"],
                                                   start_target)
        n_frames = traj_constraint["end_frame"]-start_idx + 1
        arc_length = 0.0
        self.set_pose_from_frame(motion_vector.frames[start_idx])
        prev_position = self.pose.evaluate_position(traj_constraint["joint_name"])
        for idx in range(n_frames):
            keyframe = start_idx+idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            current_position = self.pose.evaluate_position(traj_constraint["joint_name"])
            arc_length += np.linalg.norm(prev_position-current_position)
            prev_position = current_position
            if arc_length >= trajectory.full_arc_length:
                break
            target = trajectory.query_point_by_absolute_arc_length(arc_length)

            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._ik.modify_pose(traj_constraint["joint_name"], target)
                iter_counter += 1
            error_sum += error
            motion_vector.frames[keyframe] = self.pose.get_vector()

        self._create_transition_for_frame_range(motion_vector.frames, start_idx, keyframe-1, self.pose.free_joints_map[traj_constraint["joint_name"]])
        return error_sum

    def _find_corresponding_frame(self, motion_vector, start_idx, end_idx, target_joint, target_position):
        closest_start_frame = copy(start_idx)
        min_error = np.inf
        n_frames = end_idx - start_idx
        for idx in range(n_frames):
            keyframe = start_idx + idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            position = self.pose.evaluate_position(target_joint)
            error = np.linalg.norm(position - target_position)
            if error <= min_error:
                min_error = error
                closest_start_frame = keyframe
        return closest_start_frame

    def fill_rotate_events(self, motion_vector):
        for keyframe in list(motion_vector.keyframe_event_list.keyframe_events_dict["events"].keys()):
            keyframe = int(keyframe)
            for event in motion_vector.keyframe_event_list.keyframe_events_dict["events"][keyframe]:
                if event["event"] == "rotate":
                    self.fill_rotate_event(motion_vector, event)

    def fill_rotate_event(self, motion_vector, event):
        joint_name = event["parameters"]["joint"]
        orientation = event["parameters"]["globalOrientation"]
        place_keyframe = event["parameters"]["referenceKeyframe"]
        frames = motion_vector.frames[place_keyframe]
        # compare delta with global hand orientation
        joint_orientation = motion_vector.skeleton.nodes[joint_name].get_global_matrix(frames)
        joint_orientation[:3, 3] = [0, 0, 0]
        orientation_constraint = quaternion_matrix(orientation)
        delta_orientation = np.dot(np.linalg.inv(joint_orientation), orientation_constraint)
        euler = np.degrees(euler_from_matrix(delta_orientation))
        # convert to CAD coordinate system
        event["parameters"]["relativeOrientation"] = [euler[0], -euler[2], euler[1]]

    def generate_zero_frame(self):
        n_dims = len(self.skeleton.animated_joints) * 4 + 3
        zero_frame = np.zeros(n_dims)
        for j in range(len(self.skeleton.animated_joints)):
            o = j * 4 + 3
            zero_frame[o:o + 4] = [1, 0, 0, 0]
        return zero_frame

    def generate_delta_frames(self, frames, constraints, influence_range=40):
        n_frames = frames.shape[0]
        zero_frame = self.generate_zero_frame()
        constrained_frames = list(constraints.keys())
        delta_frames = collections.OrderedDict()
        delta_frames[0] = zero_frame
        delta_frames[1] = zero_frame
        for f in range(0, n_frames, influence_range):
            delta_frames[f] = zero_frame
            delta_frames[f+1] = zero_frame
        delta_frames[n_frames - 2] = zero_frame
        delta_frames[n_frames - 1] = zero_frame
        for frame_idx, frame_constraints in constraints.items():
            # delete zero frames in range around constraint
            start = max(frame_idx - influence_range, min(frame_idx, 2))
            end = min(frame_idx + influence_range, max(frame_idx, n_frames - 2))
            for i in range(start, end):
                if i in delta_frames and i not in constrained_frames:
                    del delta_frames[i]

            frame_constraints = list(frame_constraints.values())
            exp_frame = self._ik_exp.run(frames[frame_idx], frame_constraints)

            n_dims = len(self.skeleton.animated_joints) * 4 + 3
            delta_frames[frame_idx] = np.zeros(n_dims)
            delta_frames[frame_idx][3:] = convert_exp_frame_to_quat_frame(self.skeleton, exp_frame)
        delta_frames = collections.OrderedDict(sorted(delta_frames.items(), key=lambda x: x[0]))
        return list(delta_frames.keys()), np.array(list(delta_frames.values()))

    def modify_motion_vector2(self, motion_vector, plot=False):
        motion_vector.frames = self.edit_motion_using_displacement_map(motion_vector.frames, motion_vector.ik_constraints, plot=plot)
        self.apply_orientation_constraints(motion_vector.frames, motion_vector.ik_constraints)

    def edit_motion_using_displacement_map(self, frames, constraints, influence_range=40, plot=False):
        """ References
                Witkin and Popovic: Motion Warping, 1995.
                Bruderlin and Williams: Motion Signal Processing, 1995.
                Lee and Shin: A Hierarchical Approach to Interactive Motion Editing for Human-like Figures, 1999.
        """
        n_frames = len(frames)
        times = list(range(n_frames))
        d_times, delta_frames = self.generate_delta_frames(frames, constraints, influence_range)
        d_curve = CubicMotionSpline.fit_frames(self.skeleton, d_times, delta_frames)
        if plot:
            t = np.linspace(0, n_frames - 1, num=100, endpoint=True)
            d_curve.plot(t)
        new_frames = []
        for t in times:
            f = add_frames(self.skeleton, frames[t], d_curve.evaluate(t))
            new_frames.append(f)
        return np.array(new_frames)

    def apply_orientation_constraints(self, frames, constraints):
        for frame_idx, frame_constraints in constraints.items():
            for joint_name, c in frame_constraints.items():
                if c.orientation is not None and self.optimize_orientation:
                    start = c.frame_idx
                    end = c.frame_idx + 1
                    if self.activate_look_at and c.look_at:
                        self._look_at_in_range(frames, c.position, start, end)
                    print("set hand orientation", c.orientation)
                    self._set_hand_orientation(frames, c.orientation, c.joint_name, c.frame_idx, start, end)

    def edit_motion_to_look_at_target(self, frames, position, start_idx, end_idx, orient_spine=False):
        spine_joint_name = self.skeleton.skeleton_model["joints"]["spine_1"]
        head_joint_name = self.skeleton.skeleton_model["joints"]["head"]
        for frame_idx in range(start_idx, end_idx):
            if orient_spine:
                frames[frame_idx] = self.skeleton.look_at(frames[frame_idx], spine_joint_name, position, n_max_iter=1, local_dir=SPINE_LOOK_AT_DIR)
            frames[frame_idx] = self.skeleton.look_at(frames[frame_idx], head_joint_name, position, n_max_iter=1, local_dir=LOOK_AT_DIR)
            fk_nodes = self.skeleton.nodes[head_joint_name].get_fk_chain_list()
            self.interpolate_around_frame(fk_nodes, frames, start_idx, self.window)
            self.interpolate_around_frame(fk_nodes, frames, end_idx, self.window)
        return frames

    def get_static_joints(self, frame_constraints):
        static_joints = set()
        for joint_name, c in frame_constraints.items():
            if c.inside_region:
                static_joints.add(joint_name)
        return static_joints

    def find_free_root_joints(self, constraints, joint_chains):
        """ check for each joint in the joint if it is free"""
        root_joints = dict()
        for c in constraints:
            root_joints[c.joint_name] = None
            for free_joint in joint_chains[c.joint_name]:
                is_free = True
                for joint_name in joint_chains:
                    if joint_name == c.joint_name:
                        continue
                    if free_joint in joint_chains[joint_name]:
                        is_free = False
                if not is_free:
                    root_joints[c.joint_name] = free_joint
                    print("set root joint for ", c.joint_name, "to", free_joint)
                    break
        return root_joints

    def edit_motion_using_ccd(self, frames, constraints, n_max_iter=100, root_joint=None):
        new_frames = np.array(frames)
        joint_chain_buffer = dict()
        n_frames = len(frames)
        prev_static_joints = set()

        region_overlaps = []
        for frame_idx, frame_constraints in constraints.items():
            constraints = []
            fk_nodes = set()
            apply_ik = False
            static_joints = self.get_static_joints(frame_constraints)
            for joint_name, c in frame_constraints.items():
                if joint_name not in joint_chain_buffer:
                    joint_fk_nodes = self.skeleton.nodes[joint_name].get_fk_chain_list()
                    joint_chain_buffer[joint_name] = joint_fk_nodes
                if c.inside_region and prev_static_joints == static_joints:
                    #print("copy parameters for", joint_name, len(joint_chain_buffer[joint_name]))
                    #copy guess from previous frame if it is part of a region

                    self.copy_joint_parameters(joint_chain_buffer[joint_name], new_frames, frame_idx - 1, frame_idx)
                else:
                    if c.orientation is not None:
                        print("use ccd on", joint_name, "at", frame_idx, " with orientation")
                    else:
                        print("use ccd on", joint_name, "at", frame_idx)
                    constraints.append(c)
                    fk_nodes.update(joint_chain_buffer[joint_name])
                    apply_ik = True
                    if c.inside_region and prev_static_joints != static_joints:
                        region_overlaps.append(frame_idx)

            if apply_ik:
                #print("find free joints at", frame_idx)
                if len(static_joints) > 0:
                    chain_end_joints = self.find_free_root_joints(constraints, joint_chain_buffer)
                elif root_joint is not None:
                    chain_end_joints = dict()
                    for c in constraints:
                        chain_end_joints[c.joint_name] = root_joint
                else:
                    chain_end_joints = None
                new_frame = self.skeleton.reach_target_positions(new_frames[frame_idx], constraints, chain_end_joints, n_max_iter=n_max_iter, verbose=False)
                new_frames[frame_idx] = new_frame

            #  interpolate outside of region constraints

            if frame_idx + 1 in constraints:
                prev_frame_unconstrained = len(constraints[frame_idx+1])==0
            else:
                prev_frame_unconstrained = True
            if frame_idx-1 in constraints:
                next_frame_unconstrained = len(constraints[frame_idx-1]) == 0
            else:
                next_frame_unconstrained = True

            outside_of_region = next_frame_unconstrained or prev_frame_unconstrained or apply_ik

            if outside_of_region and self.window > 0 and len(fk_nodes) > 0:
                #print("outside of region", list(prev_static_joints), list(static_joints))
                fk_nodes = list(fk_nodes)
                #fk_nodes = self.skeleton.animated_joints
                self.interpolate_around_frame(fk_nodes, new_frames, frame_idx, self.window)
                #new_frames = smooth_quaternion_frames(new_frames, frame_idx,
                #                                      self.window, False)


            prev_static_joints = static_joints

        for frame_idx in region_overlaps:
            #print("apply transition smoothing", frame_idx)
            new_frames = smooth_quaternion_frames(new_frames, frame_idx,
                                                  self.window, False)
        return new_frames

    def apply_carry_constraints(self, frames, constraints):
        print("generate carry constraints")
        n_frames = frames.shape[0]
        active_orientations = dict()
        for frame_idx in range(0, n_frames):
            # update active orientations
            if frame_idx in constraints:
                for joint_name, c in constraints[frame_idx].items():
                    if c.keep_orientation and c.orientation is not None:
                        active_orientations[c.joint_name] = c.orientation
                    elif c.joint_name in active_orientations:
                        active_orientations[c.joint_name] = None
                    #else:
                    #    print("no constraint on frame", frame_idx, c.keep_orientation)
            # apply active orientations
            for joint_name in active_orientations:
                if active_orientations[joint_name] is not None:
                    #print("set orientation for", joint_name, "at", frame_idx)
                    frames[frame_idx] = self.skeleton.set_joint_orientation(frames[frame_idx], joint_name, active_orientations[joint_name] )
        return frames

    def set_joint_orientation(self, joint_name, frames, start_idx, end_idx, target_orientation):
        for frame_idx in range(start_idx, end_idx):
            frames[frame_idx] = self.skeleton.set_joint_orientation(frames[frame_idx], joint_name, target_orientation)


    def copy_joint_parameters(self, nodes, frames, src_idx, dst_idx):
        for node in nodes:
            o = self.skeleton.nodes[node].quaternion_frame_index * 4 + 3
            frames[dst_idx][o:o+4] = frames[src_idx][o:o+4]

    def interpolate_around_frame(self, fk_nodes, frames, keyframe, window):
        print("interpolate around frame", keyframe)
        for node in fk_nodes:
            o = self.skeleton.nodes[node].quaternion_frame_index * 4 + 3
            indices = list(range(o,o+4))
            smooth_joints_around_transition_using_slerp(frames, indices, keyframe, window)

        #window = 1000
        #h_window = int(window / 2)
        #start_idx = max(keyframe - h_window, 0)
        #end_idx = min(keyframe + h_window, len(frames))
        #self.apply_joint_constraints(frames, start_idx, end_idx)

    def apply_joint_constraints(self, frames, start_idx, end_idx):
        #print("apply joint constraints in range", start_idx, end_idx)
        for frame_idx in range(start_idx, end_idx):
            o = 3
            for n in self.skeleton.animated_joints:
                constraint = self.skeleton.nodes[n].joint_constraint
                if constraint is not None:
                    q = np.array(frames[frame_idx][o:o+4])
                    frames[frame_idx][o:o+4] = constraint.apply(q)
                o+=4

