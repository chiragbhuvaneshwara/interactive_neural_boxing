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

        self.bone_map = data_configuration["bone_map"]
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

        ### Trajectory info of the hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_punch_labels = np.array([0] * self.n_frames_tr_win)

        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_punch_labels = np.array([0] * self.n_frames_tr_win)

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

        tr_root_pos_local = self.convert_global_to_local(self.traj_root_pos, root_position, root_rotation)
        tr_root_vels_local = self.convert_global_to_local(self.traj_root_vels, root_position, root_rotation,
                                                          arg_type='vels')
        tr_right_pos_local = self.convert_global_to_local(self.traj_right_wrist_pos, root_position, root_rotation,
                                                          arg_type='pos', arm='right')
        tr_left_pos_local = self.convert_global_to_local(self.traj_left_wrist_pos, root_position, root_rotation,
                                                         arg_type='pos', arm='left')
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
                                            # target_vel


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
                                        desired_right_punch_label, desired_left_punch_label, right_shoulder_pos,
                                        left_shoulder_pos):
        """
        Performs blending of the future trajectory for predicted trajectory info passed in.
        :param desired_right_punch_target: np.array(3), local space
        :param desired_left_punch_target: np.array(3), local space
        :param desired_left_punch_label: np.array(1)
        :param desired_right_punch_label: np.array(1)
        :param right_shoulder_pos:
        :param left_shoulder_pos:
        :return:
        """

        # TODO blend future traj points blend with the goal i.e. the punch target when the punch target is within a
        #  certain threshold distance from the shoulder/root of the character

        def _loc_to_glob(lp):
            pos = np.array(lp)
            pos = pos.reshape(1, len(pos))
            pos_g = self.convert_local_to_global(pos, arg_type='pos')
            return pos_g.ravel()

        for hand in ['left', 'right']:
            # calc_traj_labels = False
            no_punch_mode = False
            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_pos, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                traj_labels_blend = np.array(self.traj_left_punch_labels)
                desired_punch_target_in = desired_left_punch_target
                desired_punch_target = _loc_to_glob(desired_left_punch_target)
                no_punch_target = _loc_to_glob(left_shoulder_pos)
                # no_punch_target = np.array([0,0,0])
                desired_punch_label = desired_left_punch_label
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_pos, dtype=np.float64)
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                traj_labels_blend = np.array(self.traj_right_punch_labels)
                desired_punch_target_in = desired_right_punch_target
                desired_punch_target = _loc_to_glob(desired_right_punch_target)
                no_punch_target = _loc_to_glob(right_shoulder_pos)
                # no_punch_target = np.array([0,0,0])
                desired_punch_label = desired_right_punch_label

            tr_mid_idx = self.median_idx
            if np.sum(desired_punch_target_in) == 0:
                desired_punch_target = np.array(no_punch_target)
                no_punch_mode = True
                traj_labels_blend = np.zeros(traj_labels_blend.shape)
            # else:
            #     calc_traj_labels = True

            for i in range(tr_mid_idx + 1, len(traj_pos_blend)):
                # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
                scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)
                # iterates over predictions
                if not no_punch_mode:
                    traj_pos_blend[i] = utils.glm_mix(traj_pos_blend[i],
                                                      desired_punch_target, scale_pos)

                    traj_vels_blend[i] = utils.glm_mix(traj_vels_blend[i],
                                                       traj_pos_blend[i] - traj_pos_blend[i - 1], scale_pos)
                elif no_punch_mode:
                    break
                # else:
                # break
                # traj_pos_blend[i] = utils.glm_mix(traj_pos_blend[i],
                #                                   traj_pos_blend[i], scale_pos)
                #
                # traj_vels_blend[i] = utils.glm_mix(traj_vels_blend[i],
                #                                    traj_pos_blend[i] - traj_pos_blend[i - 1], scale_pos)

                # if calc_traj_labels:
                #     # traj_labels_blend[i] = 1
                #     traj_labels_blend[i] = utils.glm_mix(traj_labels_blend[i],
                #                                          desired_punch_label, scale_pos)

                if hand == 'left':
                    self.traj_left_wrist_pos[i] = traj_pos_blend[i]
                    self.traj_left_wrist_vels[i] = traj_vels_blend[i]

                elif hand == 'right':
                    self.traj_right_wrist_pos[i] = traj_pos_blend[i]
                    self.traj_right_wrist_vels[i] = traj_vels_blend[i]

            # if calc_traj_labels:
            #     pred_tr = traj_labels_blend[tr_mid_idx:]
            #     pred_tr[np.logical_and(pred_tr < 1, pred_tr > -1)] = 0
            #     traj_labels_blend[tr_mid_idx:][traj_labels_blend[tr_mid_idx:] < 1] = 0
            #     if hand == 'left':
            #         self.traj_left_punch_labels = traj_labels_blend
            #     elif hand == "right":
            #         self.traj_right_punch_labels = traj_labels_blend

    def step_forward(self, pred_root_vel, pred_fwd_dir, pred_local_wrist_vels, curr_punch_labels):
        """
        Performs a frame-step after rendering

        Arguments:
            rot_vel {np.array(4)} -- root velocity + new root direction

        Returns:
            [type_in] -- [description]
        """
        pred_root_vel = np.array(pred_root_vel)
        pred_fwd_dir = np.array(pred_fwd_dir)

        tr_mid_idx = self.median_idx

        ## current trajectory
        self.traj_root_pos[:tr_mid_idx] = self.traj_root_pos[1:tr_mid_idx + 1]
        self.traj_root_vels[:tr_mid_idx] = self.traj_root_vels[1:tr_mid_idx + 1]
        self.traj_right_wrist_pos[:tr_mid_idx] = self.traj_right_wrist_pos[1:tr_mid_idx + 1]
        self.traj_left_wrist_pos[:tr_mid_idx] = self.traj_left_wrist_pos[1:tr_mid_idx + 1]
        self.traj_right_wrist_vels[:tr_mid_idx] = self.traj_right_wrist_vels[1:tr_mid_idx + 1]
        self.traj_left_wrist_vels[:tr_mid_idx] = self.traj_left_wrist_vels[1:tr_mid_idx + 1]
        self.traj_right_punch_labels[:tr_mid_idx] = self.traj_right_punch_labels[1:tr_mid_idx + 1]
        self.traj_left_punch_labels[:tr_mid_idx] = self.traj_left_punch_labels[1:tr_mid_idx + 1]

        def _curr_frame_update(local_vel, xz_to_x0yz=False):
            local_vel = np.array(local_vel)
            if xz_to_x0yz:
                local_vel = utils.xz_to_x0yz(local_vel)
            tr_update = local_vel.reshape(1, self.n_dims)
            tr_update = self.convert_local_to_global(tr_update, arg_type='vels')
            return tr_update

        idx = self.median_idx
        root_tr_update = _curr_frame_update(pred_root_vel, xz_to_x0yz=True)
        self.traj_root_pos[idx] = self.traj_root_pos[idx] + root_tr_update
        # self.traj_root_pos[idx] = self.traj_root_pos[idx] + np.array([0, 0, 0])
        self.traj_root_vels[idx] = utils.glm_mix(self.traj_root_vels[idx], root_tr_update, 0.9)

        pred_fwd_dir = pred_fwd_dir.ravel()
        pred_fwd_dir_x, pred_fwd_dir_z = pred_fwd_dir[0], pred_fwd_dir[1]
        rotational_vel = math.atan2(pred_fwd_dir_x, pred_fwd_dir_z)
        self.traj_root_directions[idx] = utils.rot_around_z_3d(self.traj_root_directions[idx],
                                                               rotational_vel)
        self.traj_root_rotations[idx] = utils.z_angle(self.traj_root_directions[idx])

        right_wr_v, left_wr_v = pred_local_wrist_vels
        right_tr_update = _curr_frame_update(right_wr_v)
        self.traj_right_wrist_pos[idx] = self.traj_right_wrist_pos[idx] + right_tr_update
        self.traj_right_wrist_vels[idx] = right_tr_update
        # self.traj_right_punch_labels[idx] = curr_punch_labels['right']
        if curr_punch_labels['right'] == 0:
            self.traj_right_punch_labels[idx:] = curr_punch_labels['right']
        else:
            self.traj_right_punch_labels[idx] = curr_punch_labels['right']

        left_tr_update = _curr_frame_update(left_wr_v)
        self.traj_left_wrist_pos[idx] = self.traj_left_wrist_pos[idx] + left_tr_update
        self.traj_left_wrist_vels[idx] = left_tr_update
        # self.traj_left_punch_labels[idx] = curr_punch_labels['left']
        if curr_punch_labels['left'] == 0:
            self.traj_left_punch_labels[idx:] = curr_punch_labels['left']
        else:
            self.traj_left_punch_labels[idx] = curr_punch_labels['left']

    def update_from_predict(self, prediction, curr_punch_labels):
        """
        Update trajectory from prediction.
        Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y.

        Arguments:
            prediction {np.array(4 * (6 * 2))} -- vector containing the network output regarding future trajectory positions and directions
        """
        # These predictions from the neural network are only for the trajectory points considered for the neural network
        prediction = list(prediction)

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
        pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr, pred_rpunch_tr, pred_lpunch_tr, \
            = prediction

        half_pred_window = self.median_idx // self.traj_step
        pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(half_pred_window, self.n_dims - 1))
        self.traj_root_pos[self.median_idx + 1:] = self.convert_local_to_global(
            utils.xz_to_x0yz(pred_rp_tr, axis=1))
        pred_rv_tr = _smooth_predictions(pred_rv_tr.reshape(half_pred_window, self.n_dims - 1))
        self.traj_root_vels[self.median_idx + 1:] = self.convert_local_to_global(
            utils.xz_to_x0yz(pred_rv_tr, axis=1),
            arg_type='vels')

        # TODO: Trajectory root dir is probably not required since there are not trajectory predictions for it
        # TODO: similarly traj_root_rotations is probably not required since there are not trajectory predictions for it
        # pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(self.n_frames_tr_win // self.traj_step, self.n_dims - 1))
        # self.traj_root_directions[self.median_idx + 1:] = self.convert_local_to_global(
        #     utils.normalize(utils.xz_to_x0yz(pred_fwd_dir)), arg_type="dir")
        # pred_r_rot = np.array(
        #     [utils.z_angle(self.traj_directions[i]) for i in range(self.median_idx + 1, self.n_frames_tr_win)])
        # self.traj_root_rotations[self.median_idx + 1:] = pred_r_rot

        pred_rwp_tr = _smooth_predictions(pred_rwp_tr.reshape(half_pred_window, self.n_dims))
        self.traj_right_wrist_pos[self.median_idx + 1:] = self.convert_local_to_global(pred_rwp_tr, arg_type='pos',
                                                                                       arm="right")

        pred_rwv_tr = _smooth_predictions(pred_rwv_tr.reshape(half_pred_window, self.n_dims))
        self.traj_right_wrist_vels[self.median_idx + 1:] = self.convert_local_to_global(pred_rwv_tr, arg_type='vels',
                                                                                        arm="right")

        # if curr_punch_labels["right"] != 0:
        pred_rpunch_tr = _smooth_predictions(pred_rpunch_tr.reshape(half_pred_window, 1))
        pred_rpunch_tr[pred_rpunch_tr < 1] = 0
        self.traj_right_punch_labels[self.median_idx + 1:] = pred_rpunch_tr.ravel()

        # Wrist positions contain all 3 components
        pred_lwp_tr = _smooth_predictions(pred_lwp_tr.reshape(half_pred_window, self.n_dims))
        self.traj_left_wrist_pos[self.median_idx + 1:] = self.convert_local_to_global(pred_lwp_tr, arg_type='pos',
                                                                                      arm="left")
        pred_lwv_tr = _smooth_predictions(pred_lwv_tr.reshape(half_pred_window, self.n_dims))
        self.traj_left_wrist_vels[self.median_idx + 1:] = self.convert_local_to_global(pred_lwv_tr, arg_type='vels',
                                                                                       arm="left")

        # if curr_punch_labels["left"] != 0:
        pred_lpunch_tr = _smooth_predictions(pred_lpunch_tr.reshape(half_pred_window, 1))
        pred_lpunch_tr[pred_lpunch_tr < 1] = 0
        self.traj_left_punch_labels[self.median_idx + 1:] = pred_lpunch_tr.ravel()

    def correct_foot_sliding(self, foot_sliding):
        self.traj_root_pos[self.median_idx] += foot_sliding

    def convert_global_to_local(self, arr_in, root_pos, root_rot, arg_type='pos', arm=None):
        arr_copy = np.array(arr_in)
        root_pos_copy = np.array(root_pos)
        root_rot_copy = np.array(root_rot)

        if arm == "left" or arm == "right":
            root_pos_copy[1] = 0

        for i in range(len(arr_copy)):
            curr_point = arr_copy[i]
            if arg_type == 'pos':
                curr_point -= root_pos_copy
                # arr_copy[i] -= root_pos

            # arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], root_rot, inverse=True)
            curr_point = utils.rot_around_z_3d(curr_point, root_rot_copy, inverse=True)
            # curr_point = utils.rot_around_z_3d(curr_point, root_rot)
            arr_copy[i] = curr_point

        for i in range(len(arr_copy)):
            curr_point = arr_copy[i]
            # if arg_type == 'pos':
            #     if arm == 'left':
            #         curr_point -= arr_copy[self.median_idx]
            #         # arr_copy[i] -= arr_copy[self.median_idx]
            #     elif arm == 'right':
            #         curr_point -= arr_copy[self.median_idx]
            #         # arr_copy[i] -= arr_copy[self.median_idx]

            ####
            arr_copy[i] = curr_point

        return arr_copy

    def convert_local_to_global(self, arr_in, arg_type='pos', arm=None):
        arr_copy = np.array(arr_in)

        # Info at mid trajectory is info at current frame
        root_pos = np.array(self.traj_root_pos[self.median_idx])
        root_rot = np.array(self.traj_root_rotations[self.median_idx])
        if arm == "left" or arm == "right":
            root_pos[1] = 0

        for i in range(len(arr_copy)):
            # if arg_type == 'pos':
            # if arm == 'left':
            #     #TEMPORARY add middle of array from local info
            #     arr_copy[i] = arr_copy[i] + arr_copy[0]
            # elif arm == 'right':
            #     arr_copy[i] = arr_copy[i] + arr_copy[0]

            # arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], root_rot)
            # arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], -root_rot)
            arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], root_rot)

            if arg_type == 'pos':
                arr_copy[i] = arr_copy[i] + root_pos

        return arr_copy

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
        self.traj_right_punch_labels = np.array([0] * self.n_frames_tr_win)

        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_punch_labels = np.array([0] * self.n_frames_tr_win)
