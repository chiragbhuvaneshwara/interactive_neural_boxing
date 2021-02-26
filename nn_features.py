import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
from mosi_utils_anim_t.animation_data.utils import convert_euler_frames_to_cartesian_frames, quaternion_from_matrix, \
    quaternion_matrix
from mosi_utils_anim_t.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim_t.animation_data.quaternion import Quaternion
import inspect


# TODO Cleanup
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    reqd_var_name = ''
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            reqd_var_name = var_name
    return reqd_var_name


def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    rotations = []
    for dir_vec in dir_vecs:
        q = Quaternion.between(dir_vec, ref_dir)
        rotations.append(q)
    return rotations


class FeatureExtractor:
    def __init__(self, bvh_file_path, window,
                 # type="flat",
                 to_meters=1, forward_dir=np.array([0.0, 0.0, 1.0]),
                 shoulder_joints=[8, 12], hip_joints=[15, 19], fid_l=[21, 22],
                 fid_r=[17, 18],
                 head_id=6,
                 hid_l=14,
                 hid_r=10,
                 num_traj_sampling_pts=10):
        """

        This class provides functionality to preprocess raw bvh data into a deep-learning favored format.
        It does not actually transfer the data, but provides the possibilitie to create these. An additional, lightweight process_data function is required.

        Default configurations can be loaded using set_holden_parameters and set_makehuman_parameters. Other default configurations may be added later.

            :param bvh_file_path:
            :param type_hand="flat":
            :param to_meters=1:
            :param forward_dir = [0,0,1]:
            :param shoulder_joints = [10, 20] (left, right):
            :param hip_joints = [2, 27] (left, right):
            :param fid_l = [4,5] (heel, toe):
            :param fid_r = [29, 30] (heel, toe):
        """

        self.bvh_file_path = bvh_file_path
        self.__global_positions = []

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
        # self.type = type

        self.reference_skeleton = []

        self.joint_indices_dict = {}
        self.num_traj_sampling_pts = num_traj_sampling_pts
        self.traj_step = (self.window * 2 // self.num_traj_sampling_pts)

    def reset_computations(self):
        """
        Resets computation buffers (__forwards, __root_rotations, __local_positions, __local_velocities).
        Usefull, if global_rotations are changed.
        """
        self.__forwards = []
        self.__root_rotations = []
        self.__local_positions, self.__local_velocities = [], []

    def copy(self):
        """
        Produces a copy of the current handler.
        :return FeatureExtractor
        """
        copy_feature_extractor = FeatureExtractor(self.bvh_file_path, self.window,
                                                  # self.type,
                                                  self.to_meters, self.__ref_dir, self.shoulder_joints,
                                                  self.hip_joints, self.foot_left, self.foot_right, self.head,
                                                  self.hand_left, self.hand_right, self.num_traj_sampling_pts)
        copy_feature_extractor.__global_positions = np.array(self.__global_positions)
        return copy_feature_extractor

    def set_neuron_parameters(self):
        """
        Sets the joint ids to axis neuron motion capture suit's skeleton
        """
        self.shoulder_joints = [36, 13]
        self.hip_joints = [4, 1]
        self.foot_left = [6, 6]
        self.foot_right = [3, 3]
        self.head = 12
        self.n_joints = 59
        self.to_meters = 100

    def set_awinda_parameters(self):
        """
        Sets the joint ids to xsens awinda motion capture suit's skeleton
        """
        self.shoulder_joints = [8, 12]
        self.hip_joints = [15, 19]
        self.foot_left = [21, 22]
        self.foot_right = [17, 18]
        self.head = 6
        self.hand_left = 14
        self.hand_right = 10
        self.to_meters = 100

    @staticmethod
    def load_punch_phase(punch_phase_csv_path, frame_rate_divisor=2, frame_rate_offset=0):

        punch_phase_df = pd.read_csv(punch_phase_csv_path, index_col=0, header=0)
        total_frames = len(punch_phase_df.index)
        rows_to_keep = [i for i in range(frame_rate_offset, total_frames, frame_rate_divisor)]

        punch_phase_df = punch_phase_df.iloc[rows_to_keep, :]
        punch_phase = punch_phase_df.values

        punch_dphase = punch_phase[1:] - punch_phase[:-1]
        punch_dphase[punch_dphase < 0] = (1.0 - punch_phase[:-1] + punch_phase[1:])[punch_dphase < 0]

        return punch_phase, punch_dphase

    def load_motion(self, frame_rate_divisor=2, frame_rate_offset=0):
        """
        Loads the bvh-file, sets the global_coordinates, n_joints and n_frames.
        Has to be called before any of the other functions are used.
            :param frame_rate_offset:
            :param frame_rate_divisor: frame-rate divisor (e.g. reducing framerate from 120 -> 60 fps)
        """
        print('Processing Clip %s' % self.bvh_file_path, frame_rate_divisor, frame_rate_offset)
        scale = 1 / self.to_meters
        bvhreader = BVHReader(self.bvh_file_path)

        joints_from_bvh = bvhreader.node_names.keys()
        joints_from_bvh = [joint for joint in joints_from_bvh if len(joint.split('_')) == 1]
        # print('The joints in the bvh in order are:')
        # [print(i, joint) for i, joint in enumerate(joints_from_bvh)]
        self.joint_indices_dict = {joint: i for i, joint in enumerate(joints_from_bvh)}

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
        sdr_l, sdr_r = self.shoulder_joints[0], self.shoulder_joints[1]
        hip_l, hip_r = self.hip_joints[0], self.hip_joints[1]
        global_positions = np.array(self.__global_positions)

        if len(self.__forwards) == 0:
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
        Computes forward directions. Results are stored internally to reduce future computation time.
            :return forward_dirs (np.array(n_frames, 3))
        """
        head_j = self.joint_indices_dict['Head']
        global_positions = np.array(self.__global_positions)

        if len(self.__dir_head) == 0:
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
        ref_dir = self.__ref_dir
        forward = self.get_forward_directions()

        if len(self.__root_rotations) == 0:
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

    def get_punch_targets(self, punch_phase, hand, space='local'):
        """
            CURRENTLY SET UP FOR TERTIARY PUNCH PHASE ONLY

            :param punch_phase: np.array(n_frame)
            :param hand: str, 'left' or 'right'
            :param space: str, 'local' or 'global' indicating the reqd coordinate space of the punch targets
                :return punch_target_array (np.array(n_frame, 3))
        """

        # TODO Replace with single get_punch_targets that returns punch targets for both left and right hands
        if hand == 'right':
            hand_joint = self.hand_right

        elif hand == 'left':
            hand_joint = self.hand_left

        if space == 'local':
            grp = np.array(self.__global_positions[:, 0])
            punch_target_array = np.array(self.__global_positions[:, hand_joint]) - grp
            root_rotations = self.get_root_rotations()
            for f in range(len(punch_phase)):
                punch_target_array[f] = root_rotations[f] * punch_target_array[f]
        else:
            punch_target_array = np.array(self.__global_positions[:, hand_joint])

        target_pos = np.zeros((punch_phase.shape[0], 3))

        i = 0
        while i < len(punch_phase):
            # TODO Check if this line is still required
            next_target = np.array([0.0, 0.0, 0.0])

            if punch_phase[i] == 0.0:
                target_pos[i] = np.array([0.0, 0.0, 0.0])

            elif punch_phase[i] == -1.0:
                # set punch target to be next target when hand is retreating i.e phase is -1
                # -1.0 is considered for the tertiary space
                # -1.0 indicates that the hand is returning back to the rest position
                next_target = []
                for j in range(i, len(punch_phase)):
                    if punch_phase[j] == 1.0:
                        for k in range(j, len(punch_phase)):
                            if punch_phase[k] == -1.0:
                                next_target = punch_target_array[k - 1]
                                target_pos[i:j] = next_target
                                break
                        i = j - 1
                        break
                    elif punch_phase[j] == 0.0:
                        next_target = np.array([0.0, 0.0, 0.0])
                        target_pos[i:j] = next_target
                        i = j - 1
                        break

                if len(next_target) == 0:
                    next_target = np.array([0.0, 0.0, 0.0])
                    target_pos[i:] = next_target

            elif punch_phase[i] == 1.0:
                next_target = []
                for j in range(i, len(punch_phase)):
                    if punch_phase[j] == -1.0:
                        next_target = punch_target_array[j - 1]
                        target_pos[i:j] = next_target
                        i = j - 1
                        break

                if len(next_target) == 0:
                    next_target = np.array([0.0, 0.0, 0.0])
                    target_pos[i:] = next_target

            i += 1

        punch_target_array = np.array(target_pos)

        return punch_target_array

    def get_root_velocity(self):
        """
        Returns root velocity in root local cartesian space.
            : return np.array(n_frames, 1, 3)
        """
        global_positions = np.array(self.__global_positions)
        root_rotations = self.get_root_rotations()
        root_velocity = (global_positions[1:, 0:1] - global_positions[:-1, 0:1]).copy()

        for i in range(self.n_frames - 1):
            root_velocity[i, 0][1] = 0
            root_velocity[i, 0] = root_rotations[i] * root_velocity[i, 0]

        return root_velocity

    # TODO replace with get_bone_velcoity with input param as self.root or self.hand_left etc
    def get_wrist_velocity(self, type_hand):
        """
        Returns root velocity in root local cartesian space.
            : return np.array(n_frames, 1, 3)
        """
        global_positions = np.array(self.__global_positions)
        root_rotations = self.get_root_rotations()

        if type_hand == 'left':
            hand_joint = self.hand_left
        elif type_hand == 'right':
            hand_joint = self.hand_right

        hand_velocity = (global_positions[1:, hand_joint:hand_joint + 1]
                         - global_positions[:-1, hand_joint:hand_joint + 1]).copy()

        for i in range(self.n_frames - 1):
            hand_velocity[i, 0] = root_rotations[i] * hand_velocity[i, 0]

        return hand_velocity

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

    #TODO Verify difference between get_rotational_velocity and get_new_forward_dirs
    def get_new_forward_dirs(self):
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

        return root_rvelocity

    def get_foot_concats(self, velfactor=np.array([0.05, 0.05])):
        """
        Performs a simple heuristical foot_step detection
            :param velfactor=np.array([0.05, 0.05])
                :return feet_l, feet_r  (np.array(n_frames, 1), dtype = np.float)
        """
        fid_l, fid_r = self.foot_left, self.foot_right
        velfactor = velfactor / self.to_meters

        global_positions = np.array(self.__global_positions)

        def __foot_contacts(fid):
            feet_l_x = (global_positions[1:, fid, 0] - global_positions[:-1, fid, 0]) ** 2
            feet_l_y = (global_positions[1:, fid, 1] - global_positions[:-1, fid, 1]) ** 2
            feet_l_z = (global_positions[1:, fid, 2] - global_positions[:-1, fid, 2]) ** 2
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)

            return feet_l

        # TODO verify functioning of __foot_contacts and remove comments
        # feet_l_x = (global_positions[1:, fid_l, 0] - global_positions[:-1, fid_l, 0]) ** 2
        # feet_l_y = (global_positions[1:, fid_l, 1] - global_positions[:-1, fid_l, 1]) ** 2
        # feet_l_z = (global_positions[1:, fid_l, 2] - global_positions[:-1, fid_l, 2]) ** 2
        # feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)
        #
        # feet_r_x = (global_positions[1:, fid_r, 0] - global_positions[:-1, fid_r, 0]) ** 2
        # feet_r_y = (global_positions[1:, fid_r, 1] - global_positions[:-1, fid_r, 1]) ** 2
        # feet_r_z = (global_positions[1:, fid_r, 2] - global_positions[:-1, fid_r, 2]) ** 2
        # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)

        feet_l = __foot_contacts(fid_l)
        feet_r = __foot_contacts(fid_r)

        return feet_l, feet_r

    def get_trajectory(self, frame, start_from=-1):
        """
        Computes the trajectory string for the input frame (12 surrounding points with a distance of 10 frames each)

        :param frame:
        :param num_sampling_pts:
        :param start_from: (integer) -1 if whole window should be considered, value if specific start frame should be considered (e.g. i+1)

        :return rootposs, rootdirs (np.array(12, 3))
        """
        global_positions = np.array(self.__global_positions)

        forward = self.get_forward_directions()
        head_directions = self.get_head_directions()

        # TODO: Setup action labels (what you call phase) as part of trajectory

        ####### Already rotated ########
        root_vel = np.squeeze(self.get_root_velocity())
        left_wrist_vel = np.squeeze(self.get_wrist_velocity(type_hand='left'))
        right_wrist_vel = np.squeeze(self.get_wrist_velocity(type_hand='right'))
        ####### Already rotated ########

        root_rotations = self.get_root_rotations()

        if start_from < 0:
            start_from = frame - self.window

        step = self.traj_step

        # #Global vals
        # rootposs = np.array(
        #     global_positions[start_from:frame + self.window:step, 0])
        # left_wrist_pos = np.array(
        #     global_positions[start_from:frame + self.window:step, self.hand_left]
        #    )
        #
        # right_wrist_pos = np.array(
        #     global_positions[start_from:frame + self.window:step, self.hand_right]
        #     )
        #
        # head_pos = np.array(
        #     global_positions[start_from:frame + self.window:step, self.head]
        #     )
        #
        # rootdirs = np.array(forward[start_from:frame + self.window:step])
        #
        # headdirs = np.array(head_directions[start_from:frame + self.window:step])
        #
        # rootvels = np.array(root_vel[start_from:frame + self.window:step])
        #
        # left_wristvels = np.array(left_wrist_vel[start_from:frame + self.window:step])
        # right_wristvels = np.array(right_wrist_vel[start_from:frame + self.window:step])

        # Local vals
        # TODO: verify if window should be same for punch and walk

        # TODO Set y to zero in root
        rootposs = np.array(
            global_positions[start_from:frame + self.window:step, 0] - global_positions[frame:frame + 1, 0])

        # Set y to zero in root
        left_wrist_pos = np.array(
            global_positions[start_from:frame + self.window:step, self.hand_left]
            - global_positions[frame:frame + 1, 0])

        right_wrist_pos = np.array(
            global_positions[start_from:frame + self.window:step, self.hand_right]
            - global_positions[frame:frame + 1, 0])

        head_pos = np.array(
            global_positions[start_from:frame + self.window:step, self.head]
            - global_positions[frame:frame + 1, 0])

        rootdirs = np.array(forward[start_from:frame + self.window:step])

        headdirs = np.array(head_directions[start_from:frame + self.window:step])

        rootvels = np.array(root_vel[start_from:frame + self.window:step])

        left_wristvels = np.array(left_wrist_vel[start_from:frame + self.window:step])
        right_wristvels = np.array(right_wrist_vel[start_from:frame + self.window:step])

        for j in range(len(rootposs)):
            # multiplying by root_rotation is rotating vectors to point to forward direction
            # by multiplying the inverse of the quaternion (taken care of internally ==> multiplication
            # handles only inverse rotation)
            rootposs[j] = root_rotations[frame] * rootposs[j]
            rootdirs[j] = root_rotations[frame] * rootdirs[j]
            left_wrist_pos[j] = root_rotations[frame] * left_wrist_pos[j]
            right_wrist_pos[j] = root_rotations[frame] * right_wrist_pos[j]
            headdirs[j] = root_rotations[frame] * headdirs[j]

        # # TODO explain why you need mid frame removal or not (Janis probably considers this to be important)
        # left_wrist_pos = left_wrist_pos - left_wrist_pos[len(left_wrist_pos) // 2]
        # right_wrist_pos = right_wrist_pos - right_wrist_pos[len(right_wrist_pos) // 2]

        return_items = [rootposs, left_wrist_pos, right_wrist_pos, head_pos, rootdirs, headdirs, rootvels,
                        left_wristvels, right_wristvels]

        keys = list(map(retrieve_name, return_items))
        return_tr_items = {k: v for k, v in zip(keys, return_items)}

        return return_tr_items
