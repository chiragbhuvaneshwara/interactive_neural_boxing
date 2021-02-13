"""author: Chirag Bhuvaneshwara """
import numpy as np
import math, time
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

        self.n_tr_samples = self.config_store['num_traj_samples']  # 10
        self.traj_step = self.config_store['traj_step']  # 5

        # 12 * 10 = 120 fps trajectory window or 10 * 5 = 50fps trajectory window
        self.n_frames_tr_win = self.n_tr_samples * self.traj_step
        self.median_idx = self.n_frames_tr_win // 2

        self.n_dims = 3
        unit_vecs = np.eye(self.n_dims, self.n_dims)
        z_axis = 2
        self.z_vec = unit_vecs[z_axis:z_axis + 1, :]
        z_vec_mod = np.delete(self.z_vec, 1, 1)  # Removing Y component

        ### Trajectory info of the root
        # Root positions contain all 3 components
        self.traj_positions = np.zeros((self.n_frames_tr_win, self.n_dims))
        # TODO Set vels to zero vector
        # self.traj_vels = np.tile(self.z_vec, (self.n_frames_tr_win, 1))
        self.traj_vels = np.zeros((self.n_frames_tr_win, self.n_dims))
        self.traj_rotations = np.zeros(self.n_frames_tr_win)
        self.traj_directions = np.zeros((self.n_frames_tr_win, self.n_dims))

        ### Trajectory info of the hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_positions = np.zeros((self.n_frames_tr_win, self.n_dims))
        # self.traj_right_wrist_vels = np.tile(self.z_vec, (self.n_frames_tr_win, 1))
        self.traj_right_wrist_vels = np.zeros((self.n_frames_tr_win, self.n_dims))

        # Wrist positions contain all 3 components
        self.traj_left_wrist_positions = np.zeros((self.n_frames_tr_win, self.n_dims))
        # self.traj_left_wrist_vels = np.tile(self.z_vec, (self.n_frames_tr_win, 1))
        self.traj_left_wrist_vels = np.zeros((self.n_frames_tr_win, self.n_dims))

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
        self.traj_vels = np.tile(self.z_vec, (self.n_frames_tr_win, 1))
        self.traj_right_wrist_positions = np.zeros(
            (self.n_frames_tr_win, self.n_dims))  # Wrist positions contain all 3 components
        self.traj_right_wrist_vels = np.tile(self.z_vec, (self.n_frames_tr_win, 1))

        self.traj_left_wrist_positions = np.zeros(
            (self.n_frames_tr_win, self.n_dims))  # Wrist positions contain all 3 components
        self.traj_left_wrist_vels = np.tile(self.z_vec, (self.n_frames_tr_win, 1))
        for idx in range(self.median_idx + 1):
            self.traj_positions[idx] = np.array(start_location)
            self.traj_rotations[self.median_idx] = np.array(start_orientation)
            self.traj_directions[self.median_idx] = np.array(start_direction)

    def get_input(self, root_position, root_rotation):
        """
        This function computes the network input vector based on the current trajectory state for the new root_position and root_rotations.

        Arguments:
            root_position {[type]} -- new root position
            root_rotation {[arg_type]} -- new root rotation
        Returns:
            np.array[10 * 2] -- trajectory positions
            np.array[10 * 2] -- trajectory velocities
            np.array[10 * 3] -- trajectory left wrist pos
            np.array[10 * 3] -- trajectory right wrist pos
            np.array[10 * 3] -- trajectory left wrist velocities
            np.array[10 * 3] -- trajectory right wrist velocities velocities
        """
        w = self.n_frames_tr_win // self.traj_step  # only every self.traj_step th frame => resulting in w = 10
        input_root_pos = np.zeros(((self.n_dims - 1) * w))
        input_root_vels = np.zeros(((self.n_dims - 1) * w))
        input_left_wrist_pos = np.zeros((self.n_dims * w))
        input_left_wrist_vels = np.zeros((self.n_dims * w))
        input_right_wrist_pos = np.zeros((self.n_dims * w))
        input_right_wrist_vels = np.zeros((self.n_dims * w))

        pos = self.convert_global_to_local(self.traj_positions, root_position, root_rotation)
        vels = self.convert_global_to_local(self.traj_vels, root_position, root_rotation, arg_type='vels')
        in_left_pos = self.convert_global_to_local(self.traj_left_wrist_positions, root_position, root_rotation,
                                                   arm='left')
        in_right_pos = self.convert_global_to_local(self.traj_right_wrist_positions, root_position, root_rotation,
                                                    arm='right')
        in_left_vels = self.convert_global_to_local(self.traj_left_wrist_vels, root_position, root_rotation,
                                                    arg_type='vels', arm='left')
        in_right_vels = self.convert_global_to_local(self.traj_right_wrist_vels, root_position, root_rotation,
                                                     arg_type='vels', arm='right')

        for i in range(0, self.n_frames_tr_win, self.traj_step):
            # root relative positions and directions of trajectories
            root_start_idx = (w * 0 + i // self.traj_step) * (self.n_dims - 1)
            input_root_pos[root_start_idx:root_start_idx + (self.n_dims - 1)] = pos[i][::2]
            input_root_vels[root_start_idx:root_start_idx + (self.n_dims - 1)] = vels[i][::2]

            wrist_start_idx = (w * 0 + i // self.traj_step) * self.n_dims
            input_left_wrist_pos[wrist_start_idx: wrist_start_idx + self.n_dims] = in_left_pos[i][:]
            input_right_wrist_pos[wrist_start_idx: wrist_start_idx + self.n_dims] = in_right_pos[i][:]

            input_left_wrist_vels[wrist_start_idx: wrist_start_idx + self.n_dims] = in_left_vels[i][:]
            input_right_wrist_vels[wrist_start_idx: wrist_start_idx + self.n_dims] = in_right_vels[i][:]

        return input_root_pos, input_root_vels, \
               input_right_wrist_pos, input_right_wrist_vels, \
               input_left_wrist_pos, input_left_wrist_vels

    def compute_future_root_trajectory(self, target_dir, target_vel):
        """
        Performs blending of the future trajectory for the next target direction and velocity.

        Arguments:
            target_dir {np.array(3)} -- Direction
            target_vel {np.array(3)} -- Velocity in local space
        """
        # a = self.get_global_root_tr()

        # computing future trajectory
        target_vel = target_vel.reshape(1, len(target_vel))
        target_vel = self.convert_local_to_global(target_vel, arg_type='vels')

        trajectory_positions_blend = np.array(self.traj_positions)
        tr_mid_idx = self.median_idx
        # print('######################################')
        for i in range(tr_mid_idx + 1, len(trajectory_positions_blend)):
            # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
            # scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * (tr_mid_idx)), self.blend_bias)
            scale_pos = 0.5

            # ith pos = i-1 th pos + combo of curr velocity and user requested velocity
            mixed_val = utils.glm_mix(self.traj_positions[i] - self.traj_positions[i - 1],
                                      target_vel, scale_pos)

            # print(mixed_val)
            trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + \
                                            target_vel
            # mixed_val

            self.traj_vels[i] = utils.glm_mix(self.traj_vels[i],
                                              target_vel, scale_pos)
            # adjust scale bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
            scale_dir = scale_pos
            self.traj_directions[i] = utils.mix_directions(self.traj_directions[i], target_dir,
                                                           scale_dir)
        # print('######################################')

        for i in range(tr_mid_idx + 1, len(trajectory_positions_blend)):
            self.traj_positions[i] = trajectory_positions_blend[i]
        print('######################################')

        # compute trajectory rotations
        for i in range(0, self.n_frames_tr_win):
            self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])

        # a = self.get_global_root_tr()

    def compute_future_wrist_trajectory(self, target_right_wrist_pos, target_right_wrist_vels,
                                        target_left_wrist_pos, target_left_wrist_vels):
        """
        Performs blending of the future trajectory for predicted trajectory info passed in.
        :param target_right_wrist_pos: local space
        :param target_right_wrist_vels: local space
        :param target_left_wrist_pos: local space
        :param target_left_wrist_vels: local space
        :return:
        """
        # computing future trajectory
        # print('######################################')
        # a = self.get_global_arm_tr()

        pred_samples_dims = self.n_tr_samples // 2

        for hand in ['left', 'right']:
            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_positions, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                target_wrist_pos = self.convert_local_to_global(
                    target_left_wrist_pos.reshape(pred_samples_dims, self.n_dims), arm=hand)
                target_wrist_vels = self.convert_local_to_global(
                    target_left_wrist_vels.reshape(pred_samples_dims, self.n_dims), arg_type='vels')
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_positions, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                target_wrist_pos = self.convert_local_to_global(
                    target_right_wrist_pos.reshape(pred_samples_dims, self.n_dims), arm=hand)
                target_wrist_vels = self.convert_local_to_global(
                    target_right_wrist_vels.reshape(pred_samples_dims, self.n_dims), arg_type='vels')

            tr_mid_idx = self.median_idx
            for i in range(tr_mid_idx + 1, len(traj_pos_blend)):
                # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
                # scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * (tr_mid_idx)), self.blend_bias)
                scale_pos = 0.5
                # print('wr ==> ', scale_pos)
                # iterates over predictions
                # target_idx = i // self.traj_step - self.traj_step
                target_idx = i // pred_samples_dims - pred_samples_dims

                traj_pos_blend[i] = utils.glm_mix(traj_pos_blend[i],
                                                  target_wrist_pos[target_idx], scale_pos)

                traj_vels_blend[i] = utils.glm_mix(traj_vels_blend[i],
                                                   target_wrist_vels[target_idx], scale_pos)

                if hand == 'left':
                    self.traj_left_wrist_positions[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]

                elif hand == 'right':
                    self.traj_right_wrist_positions[i] = traj_pos_blend[i]
                    self.traj_right_wrist_vels[i] = traj_vels_blend[i]
        # print('######################################')

        # a = self.get_global_arm_tr()

    def step_forward(self, pred_root_vel, pred_fwd_dir, wrist_vels_traj):
        """
        Performs a frame-step after rendering

        Arguments:
            rot_vel {np.array(4)} -- root velocity + new root direction

        Returns:
            [type] -- [description]
        """
        # a = self.get_global_root_tr()
        # a = self.get_global_arm_tr()

        # mix positions with velocity prediction
        # tr_mid_idx = len(self.traj_positions) // 2
        tr_mid_idx = self.median_idx
        for i in range(0, tr_mid_idx):
            self.traj_positions[i] = np.array(self.traj_positions[i + 1])
            self.traj_vels[i] = np.array(self.traj_vels[i + 1])
            self.traj_right_wrist_positions[i] = np.array(self.traj_right_wrist_positions[i + 1])
            self.traj_left_wrist_positions[i] = np.array(self.traj_left_wrist_positions[i + 1])
            self.traj_right_wrist_vels[i] = np.array(self.traj_right_wrist_vels[i + 1])
            self.traj_left_wrist_vels[i] = np.array(self.traj_left_wrist_vels[i + 1])
            self.traj_rotations[i] = np.array(self.traj_rotations[i + 1])
            self.traj_directions[i] = np.array(self.traj_directions[i + 1])

        ## current trajectory
        pred_root_vel = utils.convert_to_zero_y_3d(pred_root_vel)
        pred_root_vel = pred_root_vel.reshape(1, len(pred_root_vel))

        # pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)
        # pred_fwd_dir = pred_fwd_dir.reshape(1, len(pred_fwd_dir))

        pred_root_vel = self.convert_local_to_global(pred_root_vel, arg_type='vels')
        pred_fwd_dir = pred_fwd_dir.ravel()

        right_wr_v_tr, left_wr_v_tr = wrist_vels_traj

        ## Taking only the preds for next frame
        right_wr_v = right_wr_v_tr[:self.n_dims].reshape(1, self.n_dims)
        left_wr_v = left_wr_v_tr[:self.n_dims].reshape(1, self.n_dims)

        right_wr_v = self.convert_local_to_global(right_wr_v, arg_type='vels')
        left_wr_v = self.convert_local_to_global(left_wr_v, arg_type='vels')

        idx = self.median_idx
        # trajectory_update = utils.rot_around_z_3d(pred_root_vel.ravel(), self.traj_rotations[idx])
        trajectory_update = pred_root_vel

        pred_fwd_dir_x, pred_fwd_dir_z = pred_fwd_dir[0], pred_fwd_dir[1]
        rotational_vel = math.atan2(pred_fwd_dir_x, pred_fwd_dir_z)

        # right_tr_update = utils.rot_around_z_3d(right_wr_v.ravel(), self.traj_rotations[idx])
        # left_tr_update = utils.rot_around_z_3d(left_wr_v.ravel(), self.traj_rotations[idx])
        right_tr_update = right_wr_v
        left_tr_update = left_wr_v

        # print('tr:', self.traj_positions[idx])
        self.traj_positions[idx] = self.traj_positions[idx] + trajectory_update
        # print('tr nxt:', self.traj_positions[idx])

        self.traj_vels[idx] = utils.glm_mix(self.traj_vels[idx], pred_root_vel, 0.9)
        self.traj_directions[idx] = utils.rot_around_z_3d(self.traj_directions[idx],
                                                          rotational_vel)
        self.traj_rotations[idx] = utils.z_angle(self.traj_directions[idx])

        self.traj_left_wrist_positions[idx] = self.traj_left_wrist_positions[idx] + left_tr_update
        self.traj_left_wrist_vels[idx] = left_wr_v
        self.traj_right_wrist_positions[idx] = self.traj_right_wrist_positions[idx] + right_tr_update
        self.traj_right_wrist_vels[idx] = right_wr_v

        # a = self.get_global_root_tr()
        # a = self.get_global_arm_tr()

    def update_from_predict(self, prediction):
        """
        Update trajectory from prediction.
        Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y.

        Arguments:
            prediction {np.array(4 * (6 * 2))} -- vector containing the network output regarding future trajectory positions and directions
        """
        # a = self.get_global_root_tr()
        # a = self.get_global_arm_tr()

        prediction = list(prediction)
        pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr, pred_fwd_dir = prediction
        pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)

        # pred_rp_tr = utils.convert_to_zero_y_3d(pred_rp_tr.reshape(len(pred_rp_tr)//(self.n_dims-1), self.n_dims-1), axis=1).ravel()
        # pred_rv_tr = utils.convert_to_zero_y_3d(pred_rv_tr.reshape(len(pred_rv_tr)//(self.n_dims-1), self.n_dims-1), axis=1).ravel()
        # # pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir.reshape(len(pred_fwd_dir)//(self.n_dims-1), self.n_dims-1), axis=1).ravel()
        #
        # prediction = [pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr]
        # for i in range(len(prediction)):
        #     prediction[i] = prediction[i].reshape(len(prediction[i])//self.n_dims, self.n_dims)
        #
        #
        # pred_rp_tr = self.convert_local_to_global(prediction[0])
        # pred_rv_tr = self.convert_local_to_global(prediction[1])
        # pred_rwp_tr = self.convert_local_to_global(prediction[2], arm = 'right')
        # pred_lwp_tr = self.convert_local_to_global(prediction[3], arm = 'left')
        # pred_rwv_tr = self.convert_local_to_global(prediction[4], arg_type='vels')
        # pred_lwv_tr = self.convert_local_to_global(prediction[5], arg_type='vels')
        # # pred_fwd_dir = self.convert_local_to_global(prediction[6], arg_type='dir')
        # prediction = [pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr, pred_fwd_dir]
        #
        # pred_rp_tr = np.delete(prediction[0], 1, 1).ravel()
        # pred_rv_tr = np.delete(prediction[1], 1, 1).ravel()
        # pred_rwp_tr = prediction[2].ravel()
        # pred_lwp_tr = prediction[3].ravel()
        # pred_rwv_tr = prediction[4].ravel()
        # pred_lwv_tr = prediction[5].ravel()
        # # pred_fwd_dir = np.delete(prediction[6], 1, 1).ravel()
        # pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)

        # root_rotation = self.traj_rotations[self.median_idx]
        # root_pos = self.traj_positions[self.median_idx]

        n_d = self.n_dims
        # update future trajectory based on prediction. Future trajectory will solely depend on prediction and will be smoothed with the control signal by the next pre_render
        for i in range(self.median_idx + 1, self.n_frames_tr_win):
            weight = ((i - self.n_frames_tr_win / 2) / self.traj_step) % 1.0
            # The following is only relevant to smooth between 0 / self.traj_step and 1 / self.traj_step steps
            # , if 120 points are used
            self.traj_positions[i][0] = self.smooth_pred(i, 0, weight, pred_rp_tr[0::n_d - 1])
            self.traj_positions[i][2] = self.smooth_pred(i, 1, weight, pred_rp_tr[1::n_d - 1])
            # self.traj_positions[i] = self.convert_local_to_global(self.traj_positions[i].reshape(1, n_d)).ravel()
            self.traj_positions[i] = self.convert_local_to_global(self.traj_positions[i].reshape(1, n_d)).ravel()

            self.traj_vels[i][0] = self.smooth_pred(i, 0, weight, pred_rv_tr[0::n_d - 1])
            self.traj_vels[i][2] = self.smooth_pred(i, 1, weight, pred_rv_tr[1::n_d - 1])
            self.traj_vels[i] = self.convert_local_to_global(self.traj_vels[i].reshape(1, n_d), arg_type='vels').ravel()

            # self.traj_directions[i][0] = pred_fwd_dir[0]
            # self.traj_directions[i][2] = pred_fwd_dir[1]
            self.traj_directions[i] = utils.normalize(utils.glm_mix(self.traj_directions[i], pred_fwd_dir, 0.7))
            self.traj_directions[i] = self.convert_local_to_global(self.traj_directions[i].reshape(1, n_d)).ravel()

            self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])

            for axis in range(3):
                self.traj_right_wrist_positions[i][axis] = self.smooth_pred(i, axis, weight, pred_rwp_tr[axis::n_d])
                self.traj_left_wrist_positions[i][axis] = self.smooth_pred(i, axis, weight, pred_lwp_tr[axis::n_d])
                self.traj_right_wrist_vels[i][axis] = self.smooth_pred(i, axis, weight, pred_rwv_tr[axis::n_d])
                self.traj_left_wrist_vels[i][axis] = self.smooth_pred(i, axis, weight, pred_lwv_tr[axis::n_d])

            self.traj_right_wrist_positions[i] = self.convert_local_to_global(
                self.traj_right_wrist_positions[i].reshape(1, n_d), arm='right').ravel()
            self.traj_left_wrist_positions[i] = self.convert_local_to_global(
                self.traj_left_wrist_positions[i].reshape(1, n_d), arm='left').ravel()
            self.traj_right_wrist_vels[i] = self.convert_local_to_global(self.traj_right_wrist_vels[i].reshape(1, n_d),
                                                                         arg_type='vels', arm='right').ravel()
            self.traj_left_wrist_vels[i] = self.convert_local_to_global(self.traj_left_wrist_vels[i].reshape(1, n_d),
                                                                        arg_type='vels', arm='left').ravel()

        # a = self.get_global_root_tr()
        # a = self.get_global_arm_tr()

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
        half_pred_window = self.median_idx // self.traj_step
        ## idxs for iterating from 0 to 23 below for traj window size 12
        # pred_window_idx_1 = half_pred_window * 0 + (tr_frame // self.traj_step - half_pred_window)
        # pred_window_idx_2 = half_pred_window * 0 + (tr_frame // self.traj_step - half_pred_window) + (
        #                                     1 if tr_frame < (self.n_frames_tr_win - self.n_traj_samples + 1) else 0)

        ## idxs for iteracting from 6 to 11 below for traj window size 12
        # pred_window_idx_1 = tr_frame // self.traj_step
        # pred_window_idx_2 = (tr_frame // self.traj_step) + (
        #     1 if tr_frame < (self.n_frames_tr_win - self.n_traj_samples + 1) else 0)

        ## idxs for iteracting from 0 to 5 below for traj window size 12
        pred_window_idx_1 = tr_frame // self.traj_step - half_pred_window
        pred_window_idx_2 = (tr_frame // self.traj_step) + (
            1 if tr_frame < (self.n_frames_tr_win - self.n_tr_samples + 1) else 0) - half_pred_window
        a = prediction_axis[pred_window_idx_1]
        b = prediction_axis[pred_window_idx_2]

        new_pred = (1 - weight) * a + weight * b
        return new_pred

    def correct_foot_sliding(self, foot_sliding):
        self.traj_positions[self.median_idx] += foot_sliding

    def convert_global_to_local(self, arr_in, root_pos, root_rot, arg_type='pos', arm=None):
        arr = arr_in[:]

        for i in range(len(arr)):
            curr_point = arr[i]
            if arg_type == 'pos':
                curr_point -= root_pos
                # arr[i] -= root_pos

            # TODO
            # arr[i] = utils.rot_around_z_3d(arr[i], root_rot, inverse=True)
            curr_point = utils.rot_around_z_3d(curr_point, root_rot, inverse=True)
            # curr_point = utils.rot_around_z_3d(curr_point, root_rot)
            arr[i] = curr_point

        for i in range(len(arr)):
            curr_point = arr[i]
            # if arg_type == 'pos':
            #     if arm == 'left':
            #         curr_point -= arr[self.median_idx]
            #         #TODO traj_left is in global put to local
            #         # arr[i] -= arr[self.median_idx]
            #     elif arm == 'right':
            #         curr_point -= arr[self.median_idx]
            #         # arr[i] -= arr[self.median_idx]

            ####
            arr[i] = curr_point

        return arr

    def convert_local_to_global(self, arr_in, arg_type='pos', arm=None):
        arr = arr_in[:]

        # Info at mid trajectory is info at current frame
        root_pos = self.traj_positions[self.median_idx]
        root_rot = self.traj_rotations[self.median_idx]
        for i in range(len(arr)):
            # if arg_type == 'pos':
            # if arm == 'left':
            #     #TODO add middle of array from local info
            #     arr[i] = arr[i] + arr[0]
            # elif arm == 'right':
            #     arr[i] = arr[i] + arr[0]

            # TODO
            # arr[i] = utils.rot_around_z_3d(arr[i], root_rot)
            # arr[i] = utils.rot_around_z_3d(arr[i], -root_rot)
            arr[i] = utils.rot_around_z_3d(arr[i], root_rot)

            if arg_type == 'pos':
                arr[i] = arr[i] + root_pos

        return arr

    def getWorldPosRot(self):
        pos = np.array(self.traj_positions[self.median_idx])
        pos[1] = 0.0

        rot = self.traj_rotations[self.median_idx]
        return (pos, rot)

    def getPreviousPosRot(self):
        pos = np.array(self.traj_positions[self.median_idx - 1])
        pos[1] = 0.0

        rot = self.traj_rotations[self.median_idx - 1]
        return (pos, rot)

    def get_global_arm_tr(self):
        n_d = self.n_dims
        left_wr_tr = self.traj_left_wrist_positions[::self.traj_step]
        right_wr_tr = self.traj_right_wrist_positions[::self.traj_step]
        for i in range(self.n_frames_tr_win // self.traj_step):
            right_wr_tr[i] = self.convert_local_to_global(
                right_wr_tr[i].reshape(1, n_d), arm='right').ravel()
            left_wr_tr[i] = self.convert_local_to_global(
                left_wr_tr[i].reshape(1, n_d), arm='left').ravel()
        return right_wr_tr, left_wr_tr

    def get_global_root_tr(self):
        n_d = self.n_dims
        root_tr = self.traj_positions[::self.traj_step]
        for i in range(self.n_frames_tr_win // self.traj_step):
            root_tr[i] = self.convert_local_to_global(
                root_tr[i].reshape(1, n_d), arm='right').ravel()
        root_cur = root_tr[self.median_idx // self.traj_step]
        return root_tr, root_cur
