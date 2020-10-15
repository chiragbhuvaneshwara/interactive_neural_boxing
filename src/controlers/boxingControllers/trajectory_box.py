"""author: Janis Sprenger """
import numpy as np
import math, time
from ...nn.fc_models.fc_networks import FCNetwork
from ... import utils

DEBUG = False
DEBUG_TIMING = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)


class Trajectory:
    """
    This class contains data and functionality for local trajectory control.
    120 frames surrounding the current frame are considered.
    Trajectory positions and directions are predicted by the network and blended with the user input.
    This class manages gait information as well (as it is linked to the trajectory points).

    Returns:
        [type] -- [description]
    """

    def __init__(self, config_store):
        self.config_store = config_store

        self.n_traj_samples = self.config_store['num_traj_samples']   # 10
        self.traj_step_size = self.config_store['traj_step']    # 5

        # 12 * 10 = 120 fps trajectory window or 10 * 5 = 50fps trajectory window
        self.n_frames_tr_win = self.n_traj_samples * self.traj_step_size
        self.median_idx = self.n_frames_tr_win // 2

        self.num_coordinate_dims = 3
        unit_vecs = np.eye(self.num_coordinate_dims, self.num_coordinate_dims)
        z_axis = 2
        z_vec = unit_vecs[z_axis:z_axis + 1, :]
        z_vec_mod = np.delete(z_vec, 1, 1)  # Removing Y component

        # ## Trajectory info of the root
        self.traj_positions = np.zeros(
            # (self.n_frames_tr_win, self.num_coordinate_dims - 1))  # Root positions does not contain z axis
            (self.n_frames_tr_win, self.num_coordinate_dims))  # Root positions does contain z axis
        self.traj_vels = np.tile(z_vec, (self.n_frames_tr_win, 1))
        # self.traj_vels = np.tile(z_vec_mod, (self.n_frames_tr_win, 1))
        self.traj_rotations = np.zeros(self.n_frames_tr_win)
        self.traj_directions = np.zeros((self.n_frames_tr_win, self.num_coordinate_dims))

        # ## Trajectory info of the hand
        self.traj_right_wrist_positions = np.zeros(
            (self.n_frames_tr_win, self.num_coordinate_dims))  # Wrist positions contain all 3 components
        self.traj_right_wrist_vels = np.tile(z_vec, (self.n_frames_tr_win, 1))

        self.traj_left_wrist_positions = np.zeros(
            (self.n_frames_tr_win, self.num_coordinate_dims))  # Wrist positions contain all 3 components
        self.traj_left_wrist_vels = np.tile(z_vec, (self.n_frames_tr_win, 1))

        n_foot_joints = 2
        n_feet = 2  # left, right
        self.foot_drifting = np.zeros(n_foot_joints * n_feet)
        self.blend_bias = 2.0

    def reset(self, start_location=[0, 0, 0], start_orientation=[1, 0, 0, 0], start_direction=[0, 0, 1]):
        """
        Resets the trajectory information and thus the character to 0,0,0 pointing to 0,0,1.

        Keyword Arguments:
            start_location {list} -- [description] (default: {[0,0,0]})
            start_orientation {list} -- [description] (default: {[1,0,0,0]})
            start_direction {list} -- [description] (default: {[0,0,1]})
        """
        self.traj_positions = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames_tr_win)
        self.traj_rotations = np.array([0.0] * self.n_frames_tr_win)
        for idx in range(self.median_idx + 1):
            self.traj_positions[idx] = np.array(start_location)
            self.traj_rotations[self.median_idx] = np.array(start_orientation)
            self.traj_directions[self.median_idx] = np.array(start_direction)

        self.traj_gait_stand = np.array([0.0] * self.n_frames_tr_win)
        for idx in range(self.median_idx + 1):
            self.traj_gait_stand[idx] = 1
        self.traj_gait_walk = np.array([0.0] * self.n_frames_tr_win)
        self.traj_gait_jog = np.array([0.0] * self.n_frames_tr_win)
        self.traj_gait_back = np.array([0.0] * self.n_frames_tr_win)

    def convert_to_zero_y_3d(self, arr):
        return np.insert(arr, 1, 0)

    def get_input(self, root_position, root_rotation, n_gaits):
        """
        This function computes the network input vector based on the current trajectory state for the new root_position and root_rotations.

        Arguments:
            root_position {[type]} -- new root position
            root_rotation {[type]} -- new root rotation
            n_gaits {[type]} -- number of gaits
        Returns:
            np.array[12 * 2] -- trajectory positions
            np.array[12 * 2] -- trajectory directions
            np.array[12 * n_gaits] -- trajectory gaits
        """
        w = self.n_frames_tr_win // self.traj_step_size  # only every self.traj_step_size th frame => resulting in w = 12
        input_root_pos = np.zeros(((self.num_coordinate_dims-1) * w))
        input_root_vels = np.zeros(((self.num_coordinate_dims-1) * w))
        input_left_wrist_pos = np.zeros((self.num_coordinate_dims * w))
        input_left_wrist_vels = np.zeros((self.num_coordinate_dims * w))
        input_right_wrist_pos = np.zeros((self.num_coordinate_dims * w))
        input_right_wrist_vels = np.zeros((self.num_coordinate_dims * w))

        for i in range(0, self.n_frames_tr_win, self.traj_step_size):
            # root relative positions and directions of trajectories
            # diff_pos = self.convert_to_zero_y_3d(self.traj_positions[i]) - root_position
            # diff_pos = self.traj_positions[i] - root_position
            diff_pos = self.traj_positions[i]
            # root_pos = utils.rot_around_z_3d(diff_pos, root_rotation, inverse=True)
            root_pos = diff_pos
            root_start_idx = (w * 0 + i // self.traj_step_size)*(self.num_coordinate_dims-1)
            input_root_pos[root_start_idx:root_start_idx+(self.num_coordinate_dims-1)] = root_pos[::2]

            vels = utils.rot_around_z_3d(self.traj_vels[i], root_rotation, inverse=True)
            input_root_vels[root_start_idx:root_start_idx+(self.num_coordinate_dims-1)] = vels[::2]

            # left_wrist_pos = utils.rot_around_z_3d(self.traj_left_wrist_positions[i] - root_position, root_rotation,
            #                                        inverse=True)

            # left_wrist_pos = utils.rot_around_z_3d(self.traj_left_wrist_positions[i], root_rotation,
            #                                        inverse=True)
            # right_wrist_pos = utils.rot_around_z_3d(self.traj_right_wrist_positions[i], root_rotation,
            #                                         inverse=True)
            left_wrist_pos = self.traj_left_wrist_positions[i]
            right_wrist_pos = self.traj_right_wrist_positions[i]

            wrist_start_idx = (w * 0 + i // self.traj_step_size)*self.num_coordinate_dims
            input_left_wrist_pos[wrist_start_idx: wrist_start_idx + self.num_coordinate_dims] = left_wrist_pos[:]
            input_right_wrist_pos[wrist_start_idx: wrist_start_idx + self.num_coordinate_dims] = right_wrist_pos[:]

            # left_wrist_vels = utils.rot_around_z_3d(self.traj_left_wrist_vels[i], root_rotation,
            #                                         inverse=True)
            # right_wrist_vels = utils.rot_around_z_3d(self.traj_right_wrist_vels[i], root_rotation,
            #                                          inverse=True)
            left_wrist_vels = self.traj_left_wrist_vels[i]
            right_wrist_vels = self.traj_right_wrist_vels[i]

            input_left_wrist_vels[wrist_start_idx: wrist_start_idx + self.num_coordinate_dims] = left_wrist_vels[:]
            input_right_wrist_vels[wrist_start_idx: wrist_start_idx + self.num_coordinate_dims] = right_wrist_vels[:]

        if DEBUG:
            print("input dir: ")
            print("x pos: ", input_root_pos[0:12], "")
            print("y pos: ", input_root_pos[12:], "")
            print("x dir: ", input_root_vels[:12], "")
            print("y dir: ", input_root_vels[12:], "")

            print("\nworld coords: ")
            print(self.traj_positions.shape)
            print("x pos: ", self.traj_positions[::self.traj_step_size, 0])
            print("y pos: ", self.traj_positions[::self.traj_step_size, 2])
            print("x dir: ", self.traj_vels[::self.traj_step_size, 0])
            print("y dir: ", self.traj_vels[::self.traj_step_size, 2])

        return input_root_pos, input_root_vels, input_right_wrist_pos, input_right_wrist_vels, input_left_wrist_pos, \
               input_left_wrist_vels

    def compute_simple_future_wrist_trajectory(self, target_right_wrist_pos, target_right_wrist_vels, target_left_wrist_pos,
                                        target_left_wrist_vels):
        """
        """

        # computing future trajectory
        for hand in ['left', 'right']:
            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_positions, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                target_wrist_pos = target_left_wrist_pos.reshape(self.traj_step_size, self.num_coordinate_dims)
                target_wrist_vels = target_left_wrist_vels.reshape(self.traj_step_size, self.num_coordinate_dims)
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_positions, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                target_wrist_pos = target_right_wrist_pos.reshape(self.traj_step_size, self.num_coordinate_dims)
                target_wrist_vels = target_right_wrist_vels.reshape(self.traj_step_size, self.num_coordinate_dims)

            traj_pos_blend_copy = np.copy(traj_pos_blend)
            traj_pos_mid_idx = len(traj_pos_blend) // 2
            for i in range(traj_pos_mid_idx + 1, len(traj_pos_blend), self.traj_step_size):
            # for i in range(traj_pos_mid_idx, len(traj_pos_blend), self.traj_step_size):

                # This iterates through predictions vector which is of length self.n_traj_samples/2
                target_idx = i//self.traj_step_size - self.traj_step_size
                target_wrist_pos_idx = target_wrist_pos[target_idx]
                target_wrist_vels_idx = target_wrist_vels[target_idx]

                traj_pos_blend[i] = (traj_pos_blend[i] + target_wrist_pos_idx)/2

                traj_vels_blend[i] = (traj_vels_blend[i] + target_wrist_vels_idx)/2

                if hand == 'left':
                    self.traj_left_wrist_positions[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]

                elif hand == 'right':
                    self.traj_right_wrist_positions[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]


            # Verifying if this method functions correctly
            print((traj_pos_blend_copy-traj_pos_blend).ravel())

    def compute_future_wrist_trajectory(self, target_right_wrist_pos, target_right_wrist_vels, target_left_wrist_pos,
                                        target_left_wrist_vels):
        """
        """

        # computing future trajectory
        for hand in ['left', 'right']:
            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_positions, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                target_wrist_pos = target_left_wrist_pos.reshape(self.traj_step_size, self.num_coordinate_dims)
                target_wrist_vels = target_left_wrist_vels.reshape(self.traj_step_size, self.num_coordinate_dims)
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_positions, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                target_wrist_pos = target_right_wrist_pos.reshape(self.traj_step_size, self.num_coordinate_dims)
                target_wrist_vels = target_right_wrist_vels.reshape(self.traj_step_size, self.num_coordinate_dims)

            traj_pos_blend_copy = np.copy(traj_pos_blend)
            traj_pos_mid_idx = len(traj_pos_blend) // 2
            for i in range(traj_pos_mid_idx + 1, len(traj_pos_blend)):
            # for i in range(traj_pos_mid_idx, len(traj_pos_blend)):
                # less importance to idxs close to mid
                scale_weight = (i - traj_pos_mid_idx) / (1.0 * traj_pos_mid_idx)
                # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
                scale_pos = 1.0 - pow(1.0 - scale_weight, self.blend_bias)
                # traj_pos_diff = traj_pos_blend[i] - traj_pos_blend[i - 1]
                traj_pos_diff = traj_pos_blend[i]

                # This iterates through predictions vector which is of length self.n_traj_samples/2
                target_idx = i//self.traj_step_size - self.traj_step_size
                target_wrist_pos_idx = target_wrist_pos[target_idx]
                target_wrist_vels_idx = target_wrist_vels[target_idx]

                weighted_pos_update = utils.glm_mix(traj_pos_diff, target_wrist_pos_idx, scale_pos)
                traj_pos_blend[i] = weighted_pos_update
                # traj_pos_blend[i] = traj_pos_blend[i]

                # adjust scale bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
                scale_vel = scale_pos
                traj_vels_blend[i] = utils.mix_directions(traj_vels_blend[i], target_wrist_vels_idx, scale_vel)
                # traj_vels_blend[i] = traj_vels_blend[i]

                if hand == 'left':
                    self.traj_left_wrist_positions[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]

                elif hand == 'right':
                    self.traj_right_wrist_positions[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]

            # print('-----------------------------------------------------------------------------------')
            # print(traj_pos_blend.ravel())

            # Verifying if this method functions correctly
            # print((traj_pos_blend_copy-traj_pos_blend).ravel())

    def compute_root_future_trajectory(self, target_root_pos, target_root_vels, pred_fwd_dir):
        """
        Performs blending of the future trajectory for the next target direction and velocity.

        Arguments:
            target_dir {np.array(3)} -- Direction
            target_vel {np.array(3)} -- Velocity
        """
        # computing root future trajectory
        traj_pos_blend = np.array(self.traj_positions, dtype=np.float64)
        traj_pos_blend_copy = np.copy(traj_pos_blend)
        traj_vels_blend = np.array(self.traj_vels, dtype=np.float64)

        target_root_pos = target_root_pos.reshape(self.traj_step_size, self.num_coordinate_dims-1)
        target_root_pos = utils.convert_to_zero_y_3d(target_root_pos, axis=1)

        target_root_vels = target_root_vels.reshape(self.traj_step_size, self.num_coordinate_dims-1)
        target_root_vels = utils.convert_to_zero_y_3d(target_root_vels, axis=1)

        traj_pos_mid_idx = len(traj_pos_blend) // 2
        for i in range(traj_pos_mid_idx + 1, len(traj_pos_blend)):
        # for i in range(traj_pos_mid_idx, len(traj_pos_blend)):
            # less importance to idxs close to mid
            scale_weight = (i - traj_pos_mid_idx) / (1.0 * traj_pos_mid_idx)
            # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
            scale_pos = 1.0 - pow(1.0 - scale_weight, self.blend_bias)
            # traj_pos_diff = traj_pos_blend[i] - traj_pos_blend[i - 1]
            traj_pos_diff = traj_pos_blend[i]

            # This iterates through predictions vector which is of length self.n_traj_samples/2
            target_idx = i//self.traj_step_size - self.traj_step_size
            target_root_pos_idx = target_root_pos[target_idx]
            target_root_vels_idx = target_root_vels[target_idx]

            # Mixture of prediction pos and user specified pos (right now same as prediction pos)
            weighted_pos_update = utils.glm_mix(traj_pos_diff, target_root_pos_idx, scale_pos)
            traj_pos_blend[i] = weighted_pos_update
            # traj_pos_blend[i] = traj_pos_blend[i]

            # adjust scale bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
            scale_vel = scale_pos
            traj_vels_blend[i] = utils.mix_directions(traj_vels_blend[i], target_root_vels_idx, scale_vel)
            # traj_vels_blend[i] = traj_vels_blend[i]

            self.traj_positions[i] = traj_pos_blend[i]
            self.traj_vels[i] = traj_vels_blend[i]

        scale_dir = scale_pos
        pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)
        for i in range(traj_pos_mid_idx, len(traj_pos_blend)):
            # self.traj_directions[i] = utils.mix_directions(self.traj_directions[i], pred_fwd_dir,
            #                                                scale_dir)
            self.traj_directions[i] = self.traj_directions[i]


        # compute trajectory rotations
        for i in range(0, self.n_frames_tr_win):
            self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])

        # Verifying if this method functions correctly
        # print((traj_pos_blend_copy-traj_pos_blend).ravel())

    def compute_simple_root_future_trajectory(self, target_root_pos, target_root_vels, pred_fwd_dir):
        """
        Performs blending of the future trajectory for the next target direction and velocity.

        Arguments:
            target_dir {np.array(3)} -- Direction
            target_vel {np.array(3)} -- Velocity
        """
        # computing root future trajectory
        traj_pos_blend = np.array(self.traj_positions, dtype=np.float64)
        traj_vels_blend = np.array(self.traj_vels, dtype=np.float64)

        target_root_pos = target_root_pos.reshape(self.traj_step_size, self.num_coordinate_dims-1)
        target_root_pos = utils.convert_to_zero_y_3d(target_root_pos, axis=1)

        target_root_vels = target_root_vels.reshape(self.traj_step_size, self.num_coordinate_dims-1)
        target_root_vels = utils.convert_to_zero_y_3d(target_root_vels, axis=1)

        traj_pos_mid_idx = len(traj_pos_blend) // 2
        for i in range(traj_pos_mid_idx + 1, len(traj_pos_blend)):

            # This iterates through predictions vector which is of length self.n_traj_samples/2
            target_idx = i//self.traj_step_size - self.traj_step_size
            target_root_pos_idx = target_root_pos[target_idx]
            target_root_vels_idx = target_root_vels[target_idx]

            traj_pos_blend[i] = (traj_pos_blend[i] + target_root_pos_idx)/2

            traj_vels_blend[i] = (traj_vels_blend[i] + target_root_vels_idx)/2

            self.traj_positions[i] = traj_pos_blend[i]
            self.traj_vels[i] = traj_vels_blend[i]

        pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)
        for i in range(traj_pos_mid_idx, len(traj_pos_blend)):
            # self.traj_directions[i] = utils.mix_directions(self.traj_directions[i], pred_fwd_dir,scale_dir)
            self.traj_directions[i] = (self.traj_directions[i] + pred_fwd_dir)/abs(self.traj_directions[i] + pred_fwd_dir)

        # compute trajectory rotations
        for i in range(0, self.n_frames_tr_win):
            self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])

    def step_forward(self, pred_root_vel, pred_fwd_dir, wrist_vels_traj):
        """
        Performs a frame-step after rendering

        Arguments:
            rot_vel {np.array(4)} -- root velocity + new root direction

        Returns:
            [type] -- [description]
        """
        # mix positions with velocity prediction_axis
        traj_info = [self.traj_positions, self.traj_vels, self.traj_right_wrist_positions,
                     self.traj_left_wrist_positions, self.traj_right_wrist_vels, self.traj_left_wrist_vels,
                     self.traj_rotations, self.traj_directions]
        # for var in traj_info:
        #     for i in range(0, len(self.traj_positions) // 2):
        #         var[i] = np.array(var[i+1], dtype=np.float64)
        for i in range(0, len(self.traj_positions) // 2):
            self.traj_positions[i] = np.array(self.traj_positions[i + 1])
            self.traj_vels[i] = np.array(self.traj_vels[i+1])
            self.traj_right_wrist_positions[i] = np.array(self.traj_right_wrist_positions[i+1])
            self.traj_left_wrist_positions[i] = np.array(self.traj_left_wrist_positions[i+1])
            self.traj_right_wrist_vels[i] = np.array(self.traj_right_wrist_vels[i+1])
            self.traj_left_wrist_vels[i] = np.array(self.traj_left_wrist_vels[i+1])
            self.traj_rotations[i] = np.array(self.traj_rotations[i+1])
            self.traj_directions[i] = np.array(self.traj_directions[i+1])

        pred_root_vel = utils.convert_to_zero_y_3d(pred_root_vel)
        trajectory_update = utils.rot_around_z_3d(pred_root_vel, self.traj_rotations[self.median_idx])

        right_wr_v_tr, left_wr_v_tr = wrist_vels_traj
        right_wr_v = right_wr_v_tr[:self.num_coordinate_dims]
        left_wr_v = left_wr_v_tr[:self.num_coordinate_dims]

        right_tr_update = utils.rot_around_z_3d(right_wr_v, self.traj_rotations[self.median_idx])
        left_tr_update = utils.rot_around_z_3d(left_wr_v, self.traj_rotations[self.median_idx])

        pred_fwd_dir_x, pred_fwd_dir_z = pred_fwd_dir[0], pred_fwd_dir[1]
        rotational_vel = math.atan2(pred_fwd_dir_x, pred_fwd_dir_z)

        # TODO: Verify use of trajectory_update coz we dont have stand_amount (i.e gait)
        # TODO: handle foot drifitng (currently commented)
        self.traj_positions[self.median_idx] = self.traj_positions[
                                                   self.median_idx] + trajectory_update # + self.foot_drifting
        # TODO: foot_drifting and trajectory_update to be added to wrist positions?
        self.traj_left_wrist_positions[self.median_idx] = self.traj_left_wrist_positions[
                                                   self.median_idx] + right_tr_update # + self.foot_drifting
        self.traj_right_wrist_positions[self.median_idx] = self.traj_right_wrist_positions[
                                                   self.median_idx] + left_tr_update # + self.foot_drifting
        # TODO: Do I need to maintain trajectory directions? We seem to have decided that we don't need root direction (OneNote Seminar)
        self.traj_directions[self.median_idx] = utils.rot_around_z_3d(self.traj_directions[self.median_idx],
                                                                      rotational_vel)
        # TODO: Do I need to maintain trajectory rotation? We seem to have decided that we don't need root rotation (OneNote Seminar)
        self.traj_rotations[self.median_idx] = utils.z_angle(self.traj_directions[self.median_idx])

    def smooth_pred(self, tr_frame, axis, weight, prediction_axis):
        """
        Function returns new prediction value at tr_frame by combining consecutive pred values
        i.e at tr_frame = 25, returned value will be a combo of pred at 0 and pred at 1 of the prediction from NN which
        is containing only info about future frames.
        :param tr_frame:
        :param axis:
        :param weight:
        :param prediction_axis:
        :return:
        """
        half_pred_window = self.median_idx // self.traj_step_size
        ## idxs for iterating from 0 to 23 below for traj window size 12
        # pred_window_idx_1 = half_pred_window * 0 + (tr_frame // self.traj_step_size - half_pred_window)
        # pred_window_idx_2 = half_pred_window * 0 + (tr_frame // self.traj_step_size - half_pred_window) + (
        #                                     1 if tr_frame < (self.n_frames_tr_win - self.n_traj_samples + 1) else 0)

        ## idxs for iteracting from 6 to 11 below for traj window size 12
        # pred_window_idx_1 = tr_frame // self.traj_step_size
        # pred_window_idx_2 = (tr_frame // self.traj_step_size) + (
        #     1 if tr_frame < (self.n_frames_tr_win - self.n_traj_samples + 1) else 0)

        ## idxs for iteracting from 0 to 5 below for traj window size 12
        pred_window_idx_1 = tr_frame // self.traj_step_size - half_pred_window
        pred_window_idx_2 = (tr_frame // self.traj_step_size) + (
            1 if tr_frame < (self.n_frames_tr_win - self.n_traj_samples + 1) else 0) - half_pred_window
        a = prediction_axis[pred_window_idx_1]
        b = prediction_axis[pred_window_idx_2]

        new_pred = (1 - weight) * a + weight * b
        return new_pred

    def update_from_predict(self, prediction):
        """
        Update trajectory from prediction_axis.
        Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y.

        Arguments:
            prediction-- vector containing the network output regarding future trajectory. In case of boxing, network
            preds are rotated and root is subtracted already while training. So model preds don't have to be rotated and
            root doesn't have to be added. TODO Verify this
        """
        pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr, pred_fwd_dir = prediction
        n_3_d = self.num_coordinate_dims

        root_rotation = self.traj_rotations[self.median_idx]
        root_pos = self.traj_positions[self.median_idx]
        # update future trajectory based on prediction_axis. Future trajectory will solely depend on prediction_axis
        # and will be smoothed with the control signal by the next pre_render
        for i in range(self.median_idx + 1, self.n_frames_tr_win):
            weight = ((i - self.n_frames_tr_win / 2) / self.traj_step_size) % 1.0
            # The following is only relevant to smooth between 0 / self.traj_step_size and 1 / self.traj_step_size steps
            # , if 120 points are used
            self.traj_positions[i][0] = self.smooth_pred(i, 0, weight, pred_rp_tr[0::n_3_d-1])
            self.traj_positions[i][2] = self.smooth_pred(i, 1, weight, pred_rp_tr[1::n_3_d-1])
            # self.traj_positions[i] = utils.rot_around_z_3d(self.traj_positions[i], root_rotation) + root_pos

            self.traj_vels[i][0] = self.smooth_pred(i, 0, weight, pred_rv_tr[0::n_3_d-1])
            self.traj_vels[i][2] = self.smooth_pred(i, 1, weight, pred_rv_tr[1::n_3_d-1])
            # TODO Verify that rotation and root pos addition is not required because NN prediction space is already
            # of this manner.

            # self.traj_directions[i][0] = self.smooth_pred(i, 0, weight, pred_fwd_dir[0])
            self.traj_directions[i][0] = pred_fwd_dir[0]
            # self.traj_directions[i][2] = self.smooth_pred(i, 1, weight, pred_fwd_dir[1])
            self.traj_directions[i][2] = pred_fwd_dir[1]
            # self.traj_directions[i] = utils.rot_around_z_3d(utils.normalize(self.traj_directions[i]), root_rotation)

            self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])

            for axis in range(3):
                self.traj_right_wrist_positions[i][axis] = self.smooth_pred(i, axis, weight, pred_rwp_tr[axis::n_3_d])
                self.traj_left_wrist_positions[i][axis] = self.smooth_pred(i, axis, weight, pred_lwp_tr[axis::n_3_d])
                self.traj_right_wrist_vels[i][axis] = self.smooth_pred(i, axis, weight, pred_rwv_tr[axis::n_3_d])
                self.traj_right_wrist_vels[i][axis] = self.smooth_pred(i, axis, weight, pred_lwv_tr[axis::n_3_d])

    def correct_foot_sliding(self, foot_sliding):
        self.traj_positions[self.median_idx] += foot_sliding

    def getWorldPosRot(self):
        pos = np.array(self.traj_positions[self.median_idx])
        pos[1] = 0.0

        rot = self.traj_rotations[self.median_idx]
        return pos, rot

    def getPreviousPosRot(self):
        pos = np.array(self.traj_positions[self.median_idx - 1])
        pos[1] = 0.0

        rot = self.traj_rotations[self.median_idx - 1]
        return pos, rot
