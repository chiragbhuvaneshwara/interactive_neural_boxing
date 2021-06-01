import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters

from common.utils import retrieve_name
from data.neural_data_prep.mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames, \
    quaternion_from_matrix, quaternion_matrix
from data.neural_data_prep.mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from data.neural_data_prep.mosi_utils_anim.animation_data.quaternion import Quaternion


def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    """
    computes rotations of input dirs to input reference direction

    @param dir_vecs: a collection of direction vectors
    @param ref_dir: some reference direction like z axis
    @return: rotations to reference direction in radians
    """
    rotations = []
    for dir_vec in dir_vecs:
        q = Quaternion.between(dir_vec, ref_dir)
        rotations.append(q)
    return rotations


class FeatureExtractor:
    def __init__(self, bvh_file_path,window,
                 to_meters=1, forward_dir=np.array([0.0, 0.0, 1.0]),
                 shoulder_joints={'r': 8, 'l': 12},  # [right, left]
                 hip_joints={'r': 15, 'l': 19},  # [right, left]
                 fid_l={'a': 21, 't': 22},  # [ankle, toe]
                 fid_r={'a': 17, 't': 18},  # [ankle, toe]
                 head_id=6,
                 hid_l=14,
                 hid_r=10,
                 num_traj_sampling_pts=10):
        """
        This class provides functionality to preprocess raw bvh data into a Neural Network favored format.
        It does not actually transfer the data to a NN model, but provides the possibilities to create these. An
        additional, lightweight process_data function is required.
        Default configuration has been set to xsens awinda skeleton.
        Configurations can be loaded using set_awinda_parameters and set_neuron_parameters (not tested
        in pipeline). Other configurations may be added later.

        :param bvh_file_path: path to motion data in bvh format
        :param window: trajectory window for both walking and punching motions
        :param to_meters=1: to convert BVH data to proper scaling i.e height of char ~1.7m and all motions in this scale
        :param forward_dir = [0,0,1]: z axis
        :param shoulder_joints = {'r': 8, 'l': 12},  # [right, left]
        :param hip_joints = {'r': 15, 'l': 19},  # [right, left]
        :param fid_l = {'a': 21, 't': 22},  # [ankle, toe]
        :param fid_r = [{'a': 17, 't': 18},  # [ankle, toe]
        :param head_id: 6, head joint number in bvh
        :param hid_l: 14, left hand joint number
        :param hid_r: 10, right hand joint number
        :param num_traj_sampling_pts: number of traj pts to sample from specified window param
        """
        # TODO: separate trajectory window for walking and punching. Maybe even separate window for left and right
        #  punching.
        # TODO: maybe even different num_sampling_pts while changing window

        self.bvh_file_path = bvh_file_path
        self.__global_positions = []

        self.punch_labels = {}
        self.punch_labels_binary = {}
        self.delta_punch_labels = {}
        self.punch_targets = {}
        self.__bone_local_velocities = {}
        self.new_fwd_dirs = []
        self.foot_contacts = {}

        self.__dir_head = []
        self.__forwards = []
        self.__root_rotations = []
        self.__local_positions, self.__local_velocities = [], []

        self.__ref_dir = forward_dir
        self.n_frames = 0
        self.n_joints = 0

        self.shoulder_joints = shoulder_joints
        self.hip_joints = hip_joints
        self.foot_left = fid_l
        self.foot_right = fid_r
        self.head = head_id
        self.hand_left = hid_l
        self.hand_right = hid_r

        self.window = window
        self.to_meters = to_meters

        self.reference_skeleton = []

        self.joint_id_map = {}
        self.num_traj_sampling_pts = num_traj_sampling_pts
        self.traj_step = ((self.window * 2) // self.num_traj_sampling_pts)

    def reset_computations(self):
        """
        Resets computation buffers (__forwards, __root_rotations, __local_positions, __local_velocities).
        Useful, if global_rotations are changed.
        """
        # TODO Check
        self.__forwards = []
        self.__root_rotations = []
        self.__local_positions, self.__local_velocities = [], []

    def copy(self):
        """
        Produces a copy of the current handler.
        :return FeatureExtractor
        """
        # TODO Check
        copy_feature_extractor = FeatureExtractor(self.bvh_file_path, self.window,
                                                  # self.type,
                                                  self.to_meters, self.__ref_dir, self.shoulder_joints,
                                                  self.hip_joints, self.foot_left, self.foot_right, self.head,
                                                  self.hand_left, self.hand_right, self.num_traj_sampling_pts)
        copy_feature_extractor.__global_positions = np.array(self.__global_positions)
        return copy_feature_extractor

    def set_neuron_parameters(self):
        """
        Sets the joint ids to axis neuron motion capture suit's skeleton. Not yet tested.
        """
        self.shoulder_joints = {'r': 13, 'l': 36}  # TODO Check which is right and left
        self.hip_joints = {'r': 1, 'l': 4}
        self.foot_left = {'a': 6, 't': 6}
        self.foot_right = {'a': 3, 't': 3}
        self.head = 12
        self.n_joints = 59
        self.to_meters = 100

    def set_awinda_parameters(self):
        """
        Sets the joint ids to xsens awinda motion capture suit's skeleton
        """
        self.shoulder_joints = {'r': 8, 'l': 12}  # [right, left]
        self.hip_joints = {'r': 15, 'l': 19}
        self.foot_left = {'a': 21, 't': 22}  # [ankle, toe]
        self.foot_right = {'a': 17, 't': 18}
        self.head = 6
        self.hand_left = 14
        self.hand_right = 10
        self.to_meters = 100

    def load_punch_action_labels(self, punch_labels_csv_path, frame_rate_divisor=2, frame_rate_offset=0):
        """
        Helper method to load and internally store the blender generated punch labels.
        load_motionload_motion and load_punch_action_labels have to be called before any of the other functions are used.

        @param punch_labels_csv_path: string path to blender generated punch labels
        @param frame_rate_divisor: int, # if 2, Reduces fps from 120fps to 60fps (60 fps reqd. for Axis Neuron bvh)
        @param frame_rate_offset: int, offset for beginning of data to be chosen (must be synced with offset for
        load_motion method. Refer process_data() in processer.py)
        """
        punch_phase_df = pd.read_csv(punch_labels_csv_path, index_col=0, header=0)

        for hand_id in [self.hand_left, self.hand_right]:
            if hand_id == self.hand_left:
                h_col = "left punch"
            else:
                h_col = "right punch"

            hand_df = pd.DataFrame(punch_phase_df[h_col])

            total_frames = len(hand_df.index)
            rows_to_keep = [i for i in range(frame_rate_offset, total_frames, frame_rate_divisor)]

            hand_df = hand_df.iloc[rows_to_keep, :]
            hand_punch_labels = hand_df.values

            hand_delta_punch_labels = hand_punch_labels[1:] - hand_punch_labels[:-1]
            hand_delta_punch_labels[hand_delta_punch_labels < 0] = \
                (1.0 - hand_punch_labels[:-1] + hand_punch_labels[1:])[hand_delta_punch_labels < 0]

            self.punch_labels[hand_id] = hand_punch_labels

            p_labs = np.array(hand_punch_labels)
            p_labs[np.where(p_labs < 0)] = 0
            self.punch_labels_binary[hand_id] = p_labs

            self.delta_punch_labels[hand_id] = hand_delta_punch_labels

    def load_motion(self, frame_rate_divisor=2, frame_rate_offset=0, get_joint_info=False):
        """
        Loads the bvh-file, sets the global_coordinates, n_joints and n_frames and stores many values internally.
        load_motionload_motion and load_punch_action_labels have to be called before any of the other functions are used.

        :param frame_rate_offset:int, offset for beginning of data to be chosen (must be synced with offset for
        load_motion method. Refer process_data() in processer.py)
        :param frame_rate_divisor: frame-rate divisor (e.g. reducing framerate from 120 -> 60 fps)
        """
        # print('Processing Clip %s' % self.bvh_file_path, frame_rate_divisor, frame_rate_offset)
        scale = 1 / self.to_meters
        bvhreader = BVHReader(self.bvh_file_path)

        joints_from_bvh = bvhreader.node_names.keys()
        joints_from_bvh = [joint for joint in joints_from_bvh if len(joint.split('_')) == 1]

        if get_joint_info:
            print('The joints in the bvh in order are:')
            [print(i, joint) for i, joint in enumerate(joints_from_bvh)]

        self.joint_id_map = {joint: i for i, joint in enumerate(joints_from_bvh)}

        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        zero_rotations = np.zeros(bvhreader.frames.shape[1])
        zero_posture = convert_euler_frames_to_cartesian_frames(skeleton, np.array([zero_rotations]))[0]
        zero_posture[:, 0] *= -1

        self.reference_skeleton = []
        mapping = {}
        for b in skeleton.animated_joints:
            node_desc = skeleton._get_node_desc(b)
            self.reference_skeleton.append({"name": b})
            mapping[b] = int(node_desc["index"])
            self.reference_skeleton[-1]["parent"] = "" if node_desc["parent"] is None else node_desc["parent"]
            children = []
            for c in node_desc["children"]:
                if "EndSite" in c["name"]:
                    continue
                else:
                    children.append(c["name"])
            self.reference_skeleton[-1]["children"] = children
            self.reference_skeleton[-1]["index"] = node_desc["index"]
            self.reference_skeleton[-1]["position"] = zero_posture[int(node_desc["index"])].tolist()
            child_id = 0

            target_pos = np.array(zero_posture[int(node_desc["children"][child_id]["index"])])
            my_pos = np.array(self.reference_skeleton[-1]["position"])
            target_dir = (target_pos - my_pos)

            if np.linalg.norm(target_dir) < 0.0001:
                rotation = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                rotation = np.array([1.0, 0.0, 0.0, 0.0])
            self.reference_skeleton[-1]["rotation"] = rotation.tolist()

            # local rotation:
            if node_desc["parent"] is not None:
                parent_rot = np.array(self.reference_skeleton[mapping[node_desc["parent"]]]["rotation"])
            else:
                parent_rot = np.array([1.0, 0.0, 0.0, 0.0])
            inv_parent = np.linalg.inv(quaternion_matrix(parent_rot))
            loc_rot = quaternion_from_matrix(np.matmul(quaternion_matrix(rotation), inv_parent))

            self.reference_skeleton[-1]["local_rotation"] = loc_rot.tolist()

            # local position:
            loc_pos = np.array([0.0, 0.0, 0.0])
            if node_desc["parent"] is not None:
                loc_pos[1] = np.linalg.norm(my_pos - zero_posture[mapping[node_desc["parent"]]])
            self.reference_skeleton[-1]["local_position"] = loc_pos.tolist()

        cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
        global_positions = cartesian_frames * scale

        self.__global_positions = global_positions[frame_rate_offset::frame_rate_divisor]
        self.n_frames, self.n_joints, _ = self.__global_positions.shape

    def get_forward_directions(self):
        """
        Computes forward directions. Results are stored internally to reduce future computation time.

        :return forward_dirs (np.array(n_frames, 3))
        """
        if len(self.__forwards) == 0:
            sdr_l, sdr_r = self.shoulder_joints['l'], self.shoulder_joints['r']
            hip_l, hip_r = self.hip_joints['l'], self.hip_joints['r']
            global_positions = np.array(self.__global_positions)
            across = (
                    (global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
                    (global_positions[:, hip_l] - global_positions[:, hip_r]))
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

            """ Smooth Forward Direction """
            direction_filterwidth = 20
            forward = filters.gaussian_filter1d(
                np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
            self.__forwards = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        return self.__forwards

    # TODO: Verify functioning for dir head: global head pos not guaranteed to point across
    #  Is head dir reqd? Doesnt forward dir represent the same info for the data you have
    def get_head_directions(self):
        """
        Computes head facing directions. Results are stored internally to reduce future computation time.

        :return forward_dirs (np.array(n_frames, 3))
        """
        if len(self.__dir_head) == 0:
            head_j = self.joint_id_map['Head']
            global_positions = np.array(self.__global_positions)

            across = (global_positions[:, head_j])
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

            """ Smooth Forward Direction """
            direction_filterwidth = 20
            dir_head = filters.gaussian_filter1d(
                np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
            self.__dir_head = dir_head / np.sqrt((dir_head ** 2).sum(axis=-1))[..., np.newaxis]
        return self.__dir_head

    def get_root_rotations(self):
        """
        Returns root rotations. Results are stored internally to reduce future computation time.

        :return root_rotations (List(Quaternion), n_frames length)
        """
        if len(self.__root_rotations) == 0:
            ref_dir = self.__ref_dir
            forward = self.get_forward_directions()
            self.__root_rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)
        return self.__root_rotations

    def __root_local_transform(self):
        """
        Helper function to compute and store local transformations.
        """
        if len(self.__local_positions) == 0:
            local_positions = np.array(self.__global_positions)
            local_velocities = np.zeros(local_positions.shape)

            local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
            local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]

            root_rotations = self.get_root_rotations()

            for i in range(self.n_frames - 1):
                for j in range(self.n_joints):
                    local_positions[i, j] = root_rotations[i] * local_positions[i, j]

                    local_velocities[i, j] = root_rotations[i] * (
                            self.__global_positions[i + 1, j] - self.__global_positions[i, j])
            self.__local_positions = local_positions
            self.__local_velocities = local_velocities
        return self.__local_positions, self.__local_velocities

    def get_root_local_joint_positions(self):
        """
        Computes and returns root_local joint positions in cartesian space.

        :return joint positions (np.array(n_frames, n_joints, 3))
        """
        lp, _ = self.__root_local_transform()
        return lp

    def get_root_local_joint_velocities(self):
        """
        Computes and returns root_local joint velocities in cartesian space.

        :return joint velocities (np.array(n_frames, n_joints, 3))
        """
        _, lv = self.__root_local_transform()
        return lv

    def calculate_punch_targets(self, space='local'):
        """
        Based on the punch labels generated from blender for the mocap data, this method calculates the positions
        of the hands when the labels indicate punches.
        CURRENTLY SET UP FOR TERTIARY PUNCH PHASE ONLY

        :param space: str, 'local' or 'global' indicating the reqd coordinate space of the punch targets
        :return punch_target_array (np.array(n_frame, 3))
        """
        for hand_id in [self.hand_left, self.hand_right]:

            punch_labels = self.punch_labels[hand_id]

            if space == 'local':
                grp = np.array(self.__global_positions[:, 0])
                grp[:, 1] = 0  # TODO Check if punch target height is correct
                punch_target_array = np.array(self.__global_positions[:, hand_id]) - grp
                root_rotations = self.get_root_rotations()
                for f in range(len(punch_labels)):
                    punch_target_array[f] = root_rotations[f] * punch_target_array[f]
            else:
                punch_target_array = np.array(self.__global_positions[:, hand_id])

            target_pos = np.zeros((punch_labels.shape[0], 3))

            i = 0
            while i < len(punch_labels):
                # TODO Check if this line is still required
                next_target = np.array([0.0, 0.0, 0.0])

                if punch_labels[i] == 0.0:
                    target_pos[i] = np.array([0.0, 0.0, 0.0])

                elif punch_labels[i] == -1.0:
                    # set punch target to be next target when hand is retreating i.e phase is -1
                    # -1.0 is considered for the tertiary space
                    # -1.0 indicates that the hand is returning back to the rest position
                    next_target = []
                    for j in range(i, len(punch_labels)):
                        if punch_labels[j] == 1.0:
                            for k in range(j, len(punch_labels)):
                                if punch_labels[k] == -1.0:
                                    next_target = punch_target_array[k - 1]
                                    target_pos[i:j] = next_target
                                    break
                            i = j - 1
                            break
                        elif punch_labels[j] == 0.0:
                            next_target = np.array([0.0, 0.0, 0.0])
                            target_pos[i:j] = next_target
                            i = j - 1
                            break

                    if len(next_target) == 0:
                        next_target = np.array([0.0, 0.0, 0.0])
                        target_pos[i:] = next_target

                elif punch_labels[i] == 1.0:
                    next_target = []
                    for j in range(i, len(punch_labels)):
                        if punch_labels[j] == -1.0:
                            next_target = punch_target_array[j - 1]
                            target_pos[i:j] = next_target
                            i = j - 1
                            break

                    if len(next_target) == 0:
                        next_target = np.array([0.0, 0.0, 0.0])
                        target_pos[i:] = next_target

                i += 1

            punch_target_array = np.array(target_pos)
            self.punch_targets[hand_id] = punch_target_array

    def get_root_velocity(self):
        """
        Returns root velocity in root local cartesian space.

        : return np.array(n_frames, 1, 3)
        """
        if len(self.__local_velocities) == 0:
            global_positions = np.array(self.__global_positions)
            root_rotations = self.get_root_rotations()
            root_velocity = (global_positions[1:, 0:1] - global_positions[:-1, 0:1]).copy()

            for i in range(self.n_frames - 1):
                root_velocity[i, 0][1] = 0
                root_velocity[i, 0] = root_rotations[i] * root_velocity[i, 0]
        else:
            root_velocity = self.__local_velocities[:, 0:1]
            root_velocity[:, 0:1, 1] = 0

        return root_velocity

    def get_bone_velocity(self, joint_id, project_to_ground=False):
        """
        Returns root velocity in root local cartesian space.

        : return np.array(n_frames, 3)
        """
        if len(self.__local_velocities) == 0:
            if joint_id not in self.__bone_local_velocities.keys():
                global_positions = np.array(self.__global_positions)
                root_rotations = self.get_root_rotations()

                bone_velocity = (global_positions[1:, joint_id]
                                 - global_positions[:-1, joint_id]).copy()

                for i in range(self.n_frames - 1):
                    bone_velocity[i, 0] = root_rotations[i] * bone_velocity[i, 0]

                self.__bone_local_velocities[joint_id] = bone_velocity
            else:
                bone_velocity = self.__bone_local_velocities[joint_id]
        else:
            bone_velocity = self.__local_velocities[:, joint_id]

        if project_to_ground:
            bone_velocity[:, 1] = 0

        return bone_velocity

    def get_rotational_velocity(self):
        """
        Returns root rotational velocities in root local space.

        :return root_rvel (np.array(n_frames, 1, Quaternion))
        """
        root_rvelocity = np.zeros(self.n_frames - 1)
        root_rotations = self.get_root_rotations()

        for i in range(self.n_frames - 1):
            q = root_rotations[i + 1] * (-root_rotations[i])
            td = q * self.__ref_dir
            rvel = np.arctan2(td[0], td[2])
            root_rvelocity[i] = rvel

        return root_rvelocity

    # TODO Verify difference between get_rotational_velocity and get_new_forward_dirs
    def calculate_new_forward_dirs(self):
        """
        Returns the new forward direction relative to the last position.
        Alternative to rotational velocity, as this can be computed out of the new forward direction
        with np.arctan2(new_dir[0], new_dir[1])

        :return root_rvel (np.array(n_frames, 1, 2))
        """
        root_rvelocity = np.zeros((self.n_frames - 1, 2))
        root_rotations = self.get_root_rotations()

        for i in range(self.n_frames - 1):
            q = root_rotations[i + 1] * (-root_rotations[i])
            td = q * self.__ref_dir
            root_rvelocity[i] = np.array([td[0], td[2]])

        self.new_fwd_dirs = root_rvelocity

    def get_foot_concats(self, velfactor=np.array([0.05, 0.05])):
        """
        Performs a simple heuristical foot_step detection

        :param velfactor=np.array([0.05, 0.05])
        :return feet_l, feet_r  (np.array(n_frames, 1), dtype = np.float)
        """
        # TODO: see if required
        fid_l, fid_r = [self.foot_left['a'], self.foot_left['t']], [self.foot_right['a'], self.foot_right['t']]
        velfactor = velfactor / self.to_meters

        global_positions = np.array(self.__global_positions)

        def __foot_contacts(fid):
            feet_x = (global_positions[1:, fid, 0] - global_positions[:-1, fid, 0]) ** 2
            feet_y = (global_positions[1:, fid, 1] - global_positions[:-1, fid, 1]) ** 2
            feet_z = (global_positions[1:, fid, 2] - global_positions[:-1, fid, 2]) ** 2
            feet = ((feet_x + feet_y + feet_z) < velfactor).astype(np.float)

            return feet

        for fid in [fid_l, fid_r]:
            feet_contacts = __foot_contacts(fid)
            self.foot_contacts[fid[0]] = feet_contacts

    def get_trajectory(self, frame, start_from=-1):
        """
        Computes the trajectory for the input frame (self.num_traj_sampling_pts surrounding points with a distance of
        self.traj_step frames each)

        :param frame: int, current frame num as this method is called for each frame
        :param start_from: (integer) -1 if whole window should be considered, value if specific start frame should be considered (e.g. i+1)

        :return a list of trajectory items
        """
        global_positions = np.array(self.__global_positions)

        forward = self.get_forward_directions()
        # head_directions = self.get_head_directions()

        ####### Already rotated ########
        # root_vel = np.squeeze(self.get_root_velocity())
        # left_wrist_vel = np.squeeze(self.get_wrist_velocity(type_hand='left'))
        # right_wrist_vel = np.squeeze(self.get_wrist_velocity(type_hand='right'))
        root_vel = self.get_bone_velocity(0, project_to_ground=True)
        left_wrist_vel = self.get_bone_velocity(self.hand_left)
        right_wrist_vel = self.get_bone_velocity(self.hand_right)
        ####### Already rotated ########

        root_rotations = self.get_root_rotations()

        if start_from < 0:
            start_from = frame - self.window

        step = self.traj_step

        root_pos = np.array(
            global_positions[start_from:frame + self.window:step, 0] - global_positions[frame:frame + 1, 0])
        root_vels = np.array(root_vel[start_from:frame + self.window:step])
        root_dirs = np.array(forward[start_from:frame + self.window:step])

        # head_pos = np.array(
        #     global_positions[start_from:frame + self.window:step, self.head]
        #     - global_positions[frame:frame + 1, 0])
        # headdirs = np.array(head_directions[start_from:frame + self.window:step])

        # Setting y to zero in root so that traj of wrist and head are at correct height
        global_positions[:, 0, 1] = 0

        right_wrist_pos = np.array(
            global_positions[start_from:frame + self.window:step, self.hand_right]
            - global_positions[frame:frame + 1, 0])
        right_wrist_vels = np.array(right_wrist_vel[start_from:frame + self.window:step])
        # right_punch_labels = np.array(self.punch_labels[self.hand_right][start_from:frame + self.window:step])
        right_punch_labels = np.array(self.punch_labels_binary[self.hand_right][start_from:frame + self.window:step])

        left_wrist_pos = np.array(
            global_positions[start_from:frame + self.window:step, self.hand_left]
            - global_positions[frame:frame + 1, 0])
        left_wrist_vels = np.array(left_wrist_vel[start_from:frame + self.window:step])
        # left_punch_labels = np.array(self.punch_labels[self.hand_left][start_from:frame + self.window:step])
        left_punch_labels = np.array(self.punch_labels_binary[self.hand_left][start_from:frame + self.window:step])

        for j in range(len(root_pos)):
            # multiplying by root_rotation is rotating vectors to point to forward direction
            # by multiplying the inverse of the quaternion (taken care of internally ==> multiplication
            # handles only inverse rotation)
            root_pos[j] = root_rotations[frame] * root_pos[j]
            root_dirs[j] = root_rotations[frame] * root_dirs[j]
            right_wrist_pos[j] = root_rotations[frame] * right_wrist_pos[j]
            left_wrist_pos[j] = root_rotations[frame] * left_wrist_pos[j]
            # headdirs[j] = root_rotations[frame] * headdirs[j]

        # # TODO explain why you need mid frame removal or not (probably trivial and can be included)
        # right_wrist_pos = right_wrist_pos - right_wrist_pos[len(right_wrist_pos) // 2]
        # left_wrist_pos = left_wrist_pos - left_wrist_pos[len(left_wrist_pos) // 2]

        return_items = [root_pos, root_vels, root_dirs,
                        # head_pos, headdirs,
                        right_wrist_pos, right_wrist_vels, right_punch_labels,
                        left_wrist_pos, left_wrist_vels, left_punch_labels
                        ]

        keys = list(map(retrieve_name, return_items))
        return_tr_items = {k: v for k, v in zip(keys, return_items)}

        return return_tr_items
