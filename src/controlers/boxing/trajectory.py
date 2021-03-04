"""author: Chirag Bhuvaneshwara """
import numpy as np
import math
from ... import utils


class Trajectory:
    """
    This class contains data and functionality for trajectory control.
    All trajectory data inside this class should be maintained in global space.

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, data_configuration):

        self.n_tr_samples = data_configuration['num_traj_samples']  # 10
        self.traj_step = data_configuration['traj_step']  # 5

        # 12 * 10 = 120 fps trajectory window or 10 * 5 = 50fps trajectory window
        self.n_frames_tr_win = self.n_tr_samples * self.traj_step
        self.median_idx = self.n_frames_tr_win // 2

        self.n_dims = 3

        ### Trajectory info of the root
        self.traj_root_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_rotations = np.zeros(self.n_frames_tr_win)
        self.traj_root_directions = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        # TODO Set wrist positions to mean positions ==> Mabe take the positions from the mean position you have in
        #  norm

        ### Trajectory info of the hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_punch_labels = np.array([0] * self.n_frames_tr_win)

        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_punch_labels = np.array([0] * self.n_frames_tr_win)

        # TODO Cleanup

        n_foot_joints = 2
        n_feet = 2  # left, right
        self.foot_drifting = np.zeros(n_foot_joints * n_feet)
        self.blend_bias = 2.0

    def get_input(self, root_position, root_rotation):
        """
        This function computes the network input vector based on the current trajectory state for the new root_position and root_rotations.

        Arguments:
            root_position {[type_in]} -- new root position
            root_rotation {[arg_type]} -- new root rotation
        Returns:
            np.array[10 * 2] -- trajectory positions
            np.array[10 * 2] -- trajectory velocities
            np.array[10 * 3] -- trajectory left wrist pos
            np.array[10 * 3] -- trajectory right wrist pos
            np.array[10 * 3] -- trajectory left wrist velocities
            np.array[10 * 3] -- trajectory right wrist velocities velocities
        """

        # TODO remove unused code after checking functioning of traj
        # w = self.n_frames_tr_win // self.traj_step  # only every self.traj_step th frame => resulting in w = 10
        # input_root_pos = np.zeros(((self.n_dims - 1) * w))
        # input_root_vels = np.zeros(((self.n_dims - 1) * w))
        # input_left_wrist_pos = np.zeros((self.n_dims * w))
        # input_left_wrist_vels = np.zeros((self.n_dims * w))
        # input_right_wrist_pos = np.zeros((self.n_dims * w))
        # input_right_wrist_vels = np.zeros((self.n_dims * w))
        # input_left_labels = np.zeros(w)
        # input_right_labels = np.zeros(w)

        tr_root_pos_local = self.convert_global_to_local(self.traj_root_pos, root_position, root_rotation)
        tr_root_vels_local = self.convert_global_to_local(self.traj_root_vels, root_position, root_rotation,
                                                          arg_type='vels')
        # TODO Setup convert_local_to_global and vice versa to not change the y position of wrists
        tr_right_pos_local = self.convert_global_to_local(self.traj_right_wrist_pos, root_position, root_rotation,
                                                          arm='right')
        tr_left_pos_local = self.convert_global_to_local(self.traj_left_wrist_pos, root_position, root_rotation,
                                                         arm='left')
        tr_right_vels_local = self.convert_global_to_local(self.traj_right_wrist_vels, root_position, root_rotation,
                                                           arg_type='vels', arm='right')
        tr_left_vels_local = self.convert_global_to_local(self.traj_left_wrist_vels, root_position, root_rotation,
                                                          arg_type='vels', arm='left')
        tr_right_labels = self.traj_right_punch_labels
        tr_left_labels = self.traj_left_punch_labels

        # Deleting Y axis since root pos doesnt contain Y axis
        input_root_pos = np.delete(tr_root_pos_local[::self.traj_step], obj=1, axis=1).ravel()
        input_root_vels = np.delete(tr_root_vels_local[::self.traj_step], obj=1, axis=1).ravel()
        input_right_wrist_pos = tr_right_pos_local[::self.traj_step].ravel()
        input_left_wrist_pos = tr_left_pos_local[::self.traj_step].ravel()
        input_right_wrist_vels = tr_right_vels_local[::self.traj_step].ravel()
        input_left_wrist_vels = tr_left_vels_local[::self.traj_step].ravel()
        input_right_labels = tr_right_labels[::self.traj_step].ravel()
        input_left_labels = tr_left_labels[::self.traj_step].ravel()

        # for i in range(0, self.n_frames_tr_win, self.traj_step):
        #     # root relative positions and directions of trajectories
        #     root_start_idx = (w * 0 + i // self.traj_step) * (self.n_dims - 1)
        #     input_root_pos[root_start_idx:root_start_idx + (self.n_dims - 1)] = tr_root_pos_local[i][::2]
        #     input_root_vels[root_start_idx:root_start_idx + (self.n_dims - 1)] = tr_root_vels_local[i][::2]
        #
        #     wrist_start_idx = (w * 0 + i // self.traj_step) * self.n_dims
        #     input_left_wrist_pos[wrist_start_idx: wrist_start_idx + self.n_dims] = tr_left_pos_local[i][:]
        #     input_right_wrist_pos[wrist_start_idx: wrist_start_idx + self.n_dims] = tr_right_pos_local[i][:]
        #
        #     input_left_wrist_vels[wrist_start_idx: wrist_start_idx + self.n_dims] = tr_left_vels_local[i][:]
        #     input_right_wrist_vels[wrist_start_idx: wrist_start_idx + self.n_dims] = tr_right_vels_local[i][:]

        return input_root_pos, input_root_vels, \
               input_right_wrist_pos, input_left_wrist_pos, \
               input_right_wrist_vels, input_left_wrist_vels, \
               input_right_labels, input_left_labels

    def compute_future_root_trajectory(self, target_dir, target_vel):
        """
        Performs blending of the future trajectory for the next target direction and velocity.

        Arguments:
            target_dir {np.array(3)} -- Direction
            target_vel {np.array(3)} -- Velocity in local space
        """

        # computing future trajectory
        target_vel = target_vel.reshape(1, len(target_vel))
        target_vel = self.convert_local_to_global(target_vel, arg_type='vels')

        trajectory_positions_blend = np.array(self.traj_root_pos)
        tr_mid_idx = self.median_idx
        for i in range(tr_mid_idx + 1, len(trajectory_positions_blend)):
            # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
            scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)

            # ith pos = i-1 th pos + combo of curr velocity and user requested velocity
            trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + \
                                            utils.glm_mix(self.traj_root_pos[i] - self.traj_root_pos[i - 1], target_vel,
                                                          scale_pos)

            self.traj_root_vels[i] = utils.glm_mix(self.traj_root_vels[i],
                                                   target_vel, scale_pos)
            # adjust scale bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
            scale_dir = scale_pos
            self.traj_root_directions[i] = utils.mix_directions(self.traj_root_directions[i], target_dir,
                                                                scale_dir)

        for i in range(tr_mid_idx + 1, len(trajectory_positions_blend)):
            self.traj_root_pos[i] = trajectory_positions_blend[i]

        # compute trajectory rotations
        for i in range(0, self.n_frames_tr_win):
            self.traj_root_rotations[i] = utils.z_angle(self.traj_root_directions[i])

    def compute_future_wrist_trajectory(self, desired_right_punch_target, desired_left_punch_target,
                                        desired_right_punch_label, desired_left_punch_label):
        """
        Performs blending of the future trajectory for predicted trajectory info passed in.
        :param desired_right_punch_target: np.array(3), global space
        :param desired_left_punch_target: np.array(3), global space
        :param desired_left_punch_label: np.array(1)
        :param desired_right_punch_label: np.array(1)
        :return:
        """
        # TODO Instead of blending future traj points with preds, blend with the goal i.e. the punch target
        pred_samples_dims = self.n_tr_samples // 2

        for hand in ['left', 'right']:
            calc_traj_labels = False
            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_pos, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                traj_labels_blend = np.array(self.traj_left_punch_labels)
                desired_punch_target = desired_left_punch_target
                desired_punch_label = desired_left_punch_label
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_pos, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                traj_labels_blend = np.array(self.traj_right_punch_labels)
                desired_punch_target = desired_right_punch_target
                desired_punch_label = desired_right_punch_label

            tr_mid_idx = self.median_idx
            if np.sum(desired_punch_target) == 0:
                traj_labels_blend = np.zeros(traj_labels_blend.shape)
            else:
                calc_traj_labels = True

            for i in range(tr_mid_idx + 1, len(traj_pos_blend)):
                # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
                scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)
                # iterates over predictions
                # target_idx = i // self.traj_step - self.traj_step
                # target_idx = i // pred_samples_dims - pred_samples_dims
                traj_pos_blend[i] = utils.glm_mix(traj_pos_blend[i],
                                                  desired_punch_target, scale_pos)

                traj_vels_blend[i] = utils.glm_mix(traj_vels_blend[i],
                                                   traj_pos_blend[i] - traj_pos_blend[i - 1], scale_pos)

                if calc_traj_labels:
                    traj_labels_blend[i] = utils.glm_mix(traj_labels_blend[i],
                                                         desired_punch_label, scale_pos)

                if hand == 'left':
                    self.traj_left_wrist_pos[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]

                elif hand == 'right':
                    self.traj_right_wrist_pos[i] = traj_pos_blend[i]
                    self.traj_right_wrist_vels[i] = traj_vels_blend[i]

            if calc_traj_labels:
                traj_labels_blend[tr_mid_idx:][traj_labels_blend[tr_mid_idx:] < 1] = 0

    def step_forward(self, trajectory_update, pred_fwd_dir, wrist_vels_traj, punch_labels_traj):
        """
        Performs a frame-step after rendering

        Arguments:
            rot_vel {np.array(4)} -- root velocity + new root direction

        Returns:
            [type_in] -- [description]
        """
        tr_mid_idx = self.median_idx
        # for i in range(0, tr_mid_idx):
        #     self.traj_root_pos[i] = np.array(self.traj_root_pos[i + 1])
        #     self.traj_root_vels[i] = np.array(self.traj_root_vels[i + 1])
        #     self.traj_right_wrist_positions[i] = np.array(self.traj_right_wrist_positions[i + 1])
        #     self.traj_left_wrist_positions[i] = np.array(self.traj_left_wrist_positions[i + 1])
        #     self.traj_right_wrist_vels[i] = np.array(self.traj_right_wrist_vels[i + 1])
        #     self.traj_left_wrist_vels[i] = np.array(self.traj_left_wrist_vels[i + 1])
        #     self.traj_root_rotations[i] = np.array(self.traj_root_rotations[i + 1])
        #     self.traj_root_directions[i] = np.array(self.traj_root_directions[i + 1])

        ## current trajectory
        self.traj_root_pos[:tr_mid_idx] = self.traj_root_pos[1:tr_mid_idx + 1]
        self.traj_root_vels[:tr_mid_idx] = self.traj_root_vels[1:tr_mid_idx + 1]
        self.traj_right_wrist_pos[:tr_mid_idx] = self.traj_right_wrist_pos[1:tr_mid_idx + 1]
        self.traj_left_wrist_pos[:tr_mid_idx] = self.traj_left_wrist_pos[1:tr_mid_idx + 1]
        self.traj_right_wrist_vels[:tr_mid_idx] = self.traj_right_wrist_vels[1:tr_mid_idx + 1]
        self.traj_left_wrist_vels[:tr_mid_idx] = self.traj_left_wrist_vels[1:tr_mid_idx + 1]
        self.traj_right_punch_labels[:tr_mid_idx] = self.traj_right_punch_labels[1:tr_mid_idx + 1]
        self.traj_left_punch_labels[:tr_mid_idx] = self.traj_left_punch_labels[1:tr_mid_idx + 1]

        trajectory_update = utils.convert_to_zero_y_3d(trajectory_update)
        trajectory_update = trajectory_update.reshape(1, len(trajectory_update))

        # pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)
        # pred_fwd_dir = pred_fwd_dir.reshape(1, len(pred_fwd_dir))

        trajectory_update = self.convert_local_to_global(trajectory_update, arg_type='vels')
        pred_fwd_dir = pred_fwd_dir.ravel()

        right_wr_v_tr, left_wr_v_tr = wrist_vels_traj

        ## Taking only the preds for next frame
        right_tr_update = right_wr_v_tr[:self.n_dims].reshape(1, self.n_dims)
        left_tr_update = left_wr_v_tr[:self.n_dims].reshape(1, self.n_dims)

        right_tr_update = self.convert_local_to_global(right_tr_update, arg_type='vels')
        left_tr_update = self.convert_local_to_global(left_tr_update, arg_type='vels')

        idx = self.median_idx

        pred_fwd_dir_x, pred_fwd_dir_z = pred_fwd_dir[0], pred_fwd_dir[1]
        rotational_vel = math.atan2(pred_fwd_dir_x, pred_fwd_dir_z)

        self.traj_root_pos[idx] = self.traj_root_pos[idx] + trajectory_update
        self.traj_root_vels[idx] = utils.glm_mix(self.traj_root_vels[idx], trajectory_update, 0.9)
        self.traj_root_directions[idx] = utils.rot_around_z_3d(self.traj_root_directions[idx],
                                                               rotational_vel)
        self.traj_root_rotations[idx] = utils.z_angle(self.traj_root_directions[idx])

        self.traj_left_wrist_pos[idx] = self.traj_left_wrist_pos[idx] + left_tr_update
        self.traj_left_wrist_vels[idx] = left_tr_update
        self.traj_right_wrist_pos[idx] = self.traj_right_wrist_pos[idx] + right_tr_update
        self.traj_right_wrist_vels[idx] = right_tr_update

    def update_from_predict(self, prediction):
        """
        Update trajectory from prediction.
        Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y.

        Arguments:
            prediction {np.array(4 * (6 * 2))} -- vector containing the network output regarding future trajectory positions and directions
        """
        # TODO predict

        prediction = list(prediction)

        # These predictions from the neural network are only for the trajectory points considered for the neural network
        # pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr, pred_rpunch_tr, pred_lpunch_tr, \
        # pred_rpunch_tr, pred_lpunch_tr, pred_fwd_dir = prediction
        # pred_fwd_dir = utils.convert_to_zero_y_3d(pred_fwd_dir)

        # n_d = self.n_dims

        def _smooth_predictions(pred_arr):
            half_pred_window = self.median_idx // self.traj_step
            combo_idxs_1 = []
            combo_idxs_2 = []
            weights = []
            for i in range(self.median_idx + 1, self.n_frames_tr_win):
                weight = ((i - self.n_frames_tr_win / 2) / self.traj_step) % 1.0
                weights.append(weight)
                combo_idxs_1.append(i // self.traj_step - half_pred_window)
                combo_idxs_2.append((i // self.traj_step) + (
                    1 if i < (self.n_frames_tr_win - self.n_tr_samples + 1) else 0) - half_pred_window)

            # weights = np.tile(np.array(weights).reshape(len(weights), 1), pred_arr.shape[1])
            weights = np.array(weights).reshape(len(weights), 1)
            a = np.array([pred_arr[i] for i in combo_idxs_1])
            b = np.array([pred_arr[i] for i in combo_idxs_2])
            part_a = np.multiply((1 - weights), a)
            part_b = np.multiply(weights, b)
            res = part_a + part_b

            return res

        # The predictions after applying _smooth_predictions will be producing entries for all trajectories
        # maintained in trajectory class
        # prediction = list(prediction)
        # prediction = [_smooth_predictions(pred_var) for pred_var in prediction]
        pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr, pred_rpunch_tr, pred_lpunch_tr, \
            = prediction

        half_pred_window = self.median_idx // self.traj_step
        pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(half_pred_window, self.n_dims - 1))
        self.traj_root_pos[self.median_idx + 1:] = self.convert_local_to_global(utils.convert_to_zero_y_3d(pred_rp_tr, axis=1))
        pred_rv_tr = _smooth_predictions(pred_rv_tr.reshape(half_pred_window, self.n_dims - 1))
        self.traj_root_vels[self.median_idx + 1:] = self.convert_local_to_global(utils.convert_to_zero_y_3d(pred_rv_tr, axis=1),
                                                                                 arg_type='vels')

        # TODO: Trajectory root is probably not required since there are not trajectory predictions for it
        # TODO: similarly traj_root_rotations is probably not required since there are not trajectory predictions for it
        # pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(self.n_frames_tr_win // self.traj_step, self.n_dims - 1))
        # self.traj_root_directions[self.median_idx + 1:] = self.convert_local_to_global(
        #     utils.normalize(utils.convert_to_zero_y_3d(pred_fwd_dir)), arg_type="dir")

        # pred_r_rot = np.array(
        #     [utils.z_angle(self.traj_directions[i]) for i in range(self.median_idx + 1, self.n_frames_tr_win)])
        # self.traj_root_rotations[self.median_idx + 1:] = pred_r_rot

        pred_rwp_tr = _smooth_predictions(pred_rwp_tr.reshape(half_pred_window, self.n_dims))
        self.traj_right_wrist_pos[self.median_idx + 1:] = self.convert_local_to_global(pred_rwp_tr)

        pred_rwv_tr = _smooth_predictions(pred_rwv_tr.reshape(half_pred_window, self.n_dims))
        self.traj_right_wrist_vels[self.median_idx + 1:] = self.convert_local_to_global(pred_rwv_tr, arg_type='vels',
                                                                                        arm="right")

        pred_rpunch_tr = _smooth_predictions(pred_rpunch_tr.reshape(half_pred_window, 1))
        pred_rpunch_tr[pred_rpunch_tr < 1] = 0
        self.traj_right_punch_labels[self.median_idx + 1:] = pred_rpunch_tr.ravel()

        # Wrist positions contain all 3 components
        pred_lwp_tr = _smooth_predictions(pred_lwp_tr.reshape(half_pred_window, self.n_dims))
        self.traj_left_wrist_pos[self.median_idx + 1:] = self.convert_local_to_global(pred_lwp_tr)
        pred_lwv_tr = _smooth_predictions(pred_lwv_tr.reshape(half_pred_window, self.n_dims))
        self.traj_left_wrist_vels[self.median_idx + 1:] = self.convert_local_to_global(pred_lwv_tr, arg_type='vels',
                                                                                       arm="left")

        pred_lpunch_tr = _smooth_predictions(pred_lpunch_tr.reshape(half_pred_window, 1))
        pred_lpunch_tr[pred_lpunch_tr < 1] = 0
        self.traj_left_punch_labels[self.median_idx + 1:] = pred_lpunch_tr.ravel()

        # update future trajectory based on prediction. Future trajectory will solely depend on prediction and will be smoothed with the control signal by the next pre_render
        # for i in range(self.median_idx + 1, self.n_frames_tr_win):
        #     weight = ((i - self.n_frames_tr_win / 2) / self.traj_step) % 1.0
        #     # The following is only relevant to smooth between 0 / self.traj_step and 1 / self.traj_step steps
        #     # , if 120 points are used
        #     self.traj_root_pos[i][0] = self.smooth_pred(i, 0, weight, pred_rp_tr[0::n_d - 1])
        #     self.traj_root_pos[i][2] = self.smooth_pred(i, 1, weight, pred_rp_tr[1::n_d - 1])
        #     # self.traj_positions[i] = self.convert_local_to_global(self.traj_positions[i].reshape(1, n_d)).ravel()
        #     self.traj_root_pos[i] = self.convert_local_to_global(self.traj_root_pos[i].reshape(1, n_d)).ravel()
        #
        #     self.traj_root_vels[i][0] = self.smooth_pred(i, 0, weight, pred_rv_tr[0::n_d - 1])
        #     self.traj_root_vels[i][2] = self.smooth_pred(i, 1, weight, pred_rv_tr[1::n_d - 1])
        #     self.traj_root_vels[i] = self.convert_local_to_global(self.traj_root_vels[i].reshape(1, n_d),
        #                                                           arg_type='vels').ravel()
        #
        #     # self.traj_directions[i][0] = pred_fwd_dir[0]
        #     # self.traj_directions[i][2] = pred_fwd_dir[1]
        #     self.traj_root_directions[i] = utils.normalize(
        #         utils.glm_mix(self.traj_root_directions[i], pred_fwd_dir, 0.7))
        #     self.traj_root_directions[i] = self.convert_local_to_global(
        #         self.traj_root_directions[i].reshape(1, n_d)).ravel()
        #
        #     self.traj_root_rotations[i] = utils.z_angle(self.traj_root_directions[i])
        #
        #     for axis in range(3):
        #         self.traj_right_wrist_pos[i][axis] = self.smooth_pred(i, axis, weight, pred_rwp_tr[axis::n_d])
        #         self.traj_left_wrist_pos[i][axis] = self.smooth_pred(i, axis, weight, pred_lwp_tr[axis::n_d])
        #         self.traj_right_wrist_vels[i][axis] = self.smooth_pred(i, axis, weight, pred_rwv_tr[axis::n_d])
        #         self.traj_left_wrist_vels[i][axis] = self.smooth_pred(i, axis, weight, pred_lwv_tr[axis::n_d])
        #
        #     self.traj_right_wrist_pos[i] = self.convert_local_to_global(
        #         self.traj_right_wrist_pos[i].reshape(1, n_d), arm='right').ravel()
        #     self.traj_left_wrist_pos[i] = self.convert_local_to_global(
        #         self.traj_left_wrist_pos[i].reshape(1, n_d), arm='left').ravel()
        #     self.traj_right_wrist_vels[i] = self.convert_local_to_global(self.traj_right_wrist_vels[i].reshape(1, n_d),
        #                                                                  arg_type='vels', arm='right').ravel()
        #     self.traj_left_wrist_vels[i] = self.convert_local_to_global(self.traj_left_wrist_vels[i].reshape(1, n_d),
        #                                                                 arg_type='vels', arm='left').ravel()

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

        ## idxs for iteracting from 0 to 5 below for traj window size 12
        pred_window_idx_1 = tr_frame // self.traj_step - half_pred_window
        pred_window_idx_2 = (tr_frame // self.traj_step) + (
            1 if tr_frame < (self.n_frames_tr_win - self.n_tr_samples + 1) else 0) - half_pred_window
        a = prediction_axis[pred_window_idx_1]
        b = prediction_axis[pred_window_idx_2]

        new_pred = (1 - weight) * a + weight * b
        return new_pred

    def correct_foot_sliding(self, foot_sliding):
        self.traj_root_pos[self.median_idx] += foot_sliding

    def convert_global_to_local(self, arr_in, root_pos, root_rot, arg_type='pos', arm=None):
        arr = arr_in[:]

        if arm == "left" or "right":
            root_pos[1] = 0

        for i in range(len(arr)):
            curr_point = arr[i]
            if arg_type == 'pos':
                curr_point -= root_pos
                # arr[i] -= root_pos

            # TODO Cleanup
            # arr[i] = utils.rot_around_z_3d(arr[i], root_rot, inverse=True)
            curr_point = utils.rot_around_z_3d(curr_point, root_rot, inverse=True)
            # curr_point = utils.rot_around_z_3d(curr_point, root_rot)
            arr[i] = curr_point

        for i in range(len(arr)):
            curr_point = arr[i]
            # if arg_type == 'pos':
            #     if arm == 'left':
            #         curr_point -= arr[self.median_idx]
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
        root_pos = self.traj_root_pos[self.median_idx]
        root_rot = self.traj_root_rotations[self.median_idx]
        if arm == "left" or "right":
            root_pos[1] = 0

        for i in range(len(arr)):
            # if arg_type == 'pos':
            # if arm == 'left':
            #     #TEMPORARY add middle of array from local info
            #     arr[i] = arr[i] + arr[0]
            # elif arm == 'right':
            #     arr[i] = arr[i] + arr[0]

            # arr[i] = utils.rot_around_z_3d(arr[i], root_rot)
            # arr[i] = utils.rot_around_z_3d(arr[i], -root_rot)
            arr[i] = utils.rot_around_z_3d(arr[i], root_rot)

            if arg_type == 'pos':
                arr[i] = arr[i] + root_pos

        return arr

    def get_world_pos_rot(self):
        pos = np.array(self.traj_root_pos[self.median_idx])
        pos[1] = 0.0

        rot = self.traj_root_rotations[self.median_idx]
        return pos, rot

    def get_previous_pos_rot(self):
        pos = np.array(self.traj_root_pos[self.median_idx - 1])
        pos[1] = 0.0

        rot = self.traj_root_rotations[self.median_idx - 1]
        return pos, rot

    def reset(self):
        """
        Resets the trajectory information and thus the character to 0,0,0 pointing to 0,0,1.

        Keyword Arguments:
            start_location {list} -- [description] (default: {[0,0,0]})
            start_orientation {list} -- [description] (default: {[1,0,0,0]})
            start_direction {list} -- [description] (default: {[0,0,1]})
        """
        self.traj_root_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_rotations = np.zeros(self.n_frames_tr_win)
        self.traj_root_directions = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)

        ### Trajectory info of the hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)

        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
