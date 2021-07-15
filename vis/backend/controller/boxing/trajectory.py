"""author: Chirag Bhuvaneshwara """
import numpy as np
import math

from vis.backend.controller import utils
from scipy.spatial.transform import Rotation


class Trajectory:
    """
    This class contains data and functionality for trajectory control.
    All trajectory data inside this class should be maintained in global space.
    """

    def __init__(self, data_configuration):
        self.bone_map = data_configuration["bone_map"]
        self.n_tr_samples = data_configuration['num_traj_samples']  # 10
        self.traj_step = data_configuration['traj_step']  # 5
        self.left_wrist_pos_avg_diff = np.array(data_configuration['left_wrist_pos_avg_diff']).ravel()
        self.right_wrist_pos_avg_diff = np.array(data_configuration['right_wrist_pos_avg_diff']).ravel()
        self.left_wrist_pos_avg_diff_gp = np.array(data_configuration['left_wrist_pos_avg_diff']).ravel()
        self.right_wrist_pos_avg_diff_gp = np.array(data_configuration['right_wrist_pos_avg_diff']).ravel()
        self.left_wrist_pos_no_punch_dist = data_configuration['left_wrist_no_punch']
        self.right_wrist_pos_no_punch_dist = data_configuration['right_wrist_no_punch']

        # 10 * 5 = 50fps trajectory window
        self.n_frames_tr_win = self.n_tr_samples * self.traj_step
        self.median_idx = self.n_frames_tr_win // 2

        self.n_dims = 3

        ### Trajectory info of the root
        self.traj_root_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_rotations = np.zeros(self.n_frames_tr_win)
        # self.traj_root_directions = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_root_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames_tr_win)

        ### Trajectory info of the right hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        # self.traj_right_punch_labels = np.array([0] * self.n_frames_tr_win)

        ### Trajectory info of the left hand
        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        # self.traj_left_punch_labels = np.array([0] * self.n_frames_tr_win)

        self.traj_reached_left_wrist = {'f': 0, 'r': 0}
        self.traj_reached_right_wrist = {'f': 0, 'r': 0}

        self.wrist_reached_left_wrist = False
        self.wrist_reached_right_wrist = False

        self.foot_drifting = np.zeros(2 * 2)  # (n_foot_joints * n_feet)
        # self.blend_bias = 2.0
        self.blend_bias = 10.0  # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
        self.punch_frames = 0

        self.pos_step_g = np.array([0, 0, 0])
        self.pos_step_rev_g = np.array([0, 0, 0])

    def get_input(self, root_position, root_rotation):
        """
        This method computes the network input vector based on the current trajectory state for the new root_position and root_rotations.

        @param: root_position {[type_in]} -- new root position in global co-ordinate space
        @param: root_rotation {[arg_type]} -- new root rotation
        @return:
        input_root_pos: np.array[10 * 2] -- trajectory positions in local co-ordinate space
        input_root_vels: np.array[10 * 2] -- trajectory velocities
        input_right_wrist_pos: np.array[10 * 3] -- trajectory right wrist pos in local co-ordinate space
        input_left_wrist_pos: np.array[10 * 3] -- trajectory left wrist pos in local co-ordinate space
        input_right_wrist_vels: np.array[10 * 3] -- trajectory right wrist velocities velocities
        input_left_wrist_vels: np.array[10 * 3] -- trajectory left wrist velocities
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

        # Deleting Y axis since root pos doesnt contain Y axis
        input_root_pos = np.delete(tr_root_pos_local[::self.traj_step], obj=1, axis=1).ravel()
        input_root_vels = np.delete(tr_root_vels_local[::self.traj_step], obj=1, axis=1).ravel()
        input_right_wrist_pos = tr_right_pos_local[::self.traj_step].ravel()
        input_left_wrist_pos = tr_left_pos_local[::self.traj_step].ravel()
        input_right_wrist_vels = tr_right_vels_local[::self.traj_step].ravel()
        input_left_wrist_vels = tr_left_vels_local[::self.traj_step].ravel()

        return input_root_pos, input_root_vels, \
               input_right_wrist_pos, input_left_wrist_pos, \
               input_right_wrist_vels, input_left_wrist_vels, \
            # input_right_labels, input_left_labels

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

    def _update_wrist_traj(self, traj_pos_blend, traj_vels_blend, traj_reached, wrist_reached, hand):
        if hand == 'left':
            self.traj_left_wrist_pos = traj_pos_blend[:]
            self.traj_left_wrist_vels = traj_vels_blend[:]
            self.wrist_reached_left_wrist = wrist_reached
            self.traj_reached_left_wrist = traj_reached

        elif hand == 'right':
            self.traj_right_wrist_pos = traj_pos_blend[:]
            self.traj_right_wrist_vels = traj_vels_blend[:]
            self.wrist_reached_right_wrist = wrist_reached
            self.traj_reached_right_wrist = traj_reached

    def compute_future_wrist_trajectory(self, desired_right_punch_target, desired_left_punch_target,
                                        desired_right_punch_label, desired_left_punch_label, right_shoulder_pos,
                                        left_shoulder_pos, right_wr_lp, left_wr_lp, root_position, root_rotation,
                                        traj_reached):
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

        def _loc_to_glob(lp, p=False):
            pos = np.array(lp)
            pos = pos.reshape(1, len(pos))
            pos_g = self.convert_local_to_global(pos, arg_type='pos')

            if p:
                root_pos = np.array(self.traj_root_pos[self.median_idx])
                root_rot = np.array(self.traj_root_rotations[self.median_idx])
                # print(root_pos, root_rot)

            return pos_g.ravel()

        def _glob_to_loc(gp, rp, rr):
            pos = np.array(gp)
            pos = pos.reshape(1, len(pos))
            pos_g = self.convert_global_to_local(pos, rp, rr, arg_type='pos')
            return pos_g.ravel()

        def _tr_update_pos_g(traj_pos, tr_mid_idx, desired_punch_target, no_punch_mode,
                             tr_reached, wrist_reached, pos_step_global, hand, fwd_motion=True, rev_motion=False):

            tr_reached = tr_reached.copy()
            if fwd_motion:
                start = tr_mid_idx + 1
                end = len(traj_pos) - tr_reached['f']
                step = 1
                threshold = 0.15
                print(hand, "fwd:", [i for i in range(start, end)])
            if rev_motion:
                # start = len(traj_pos) - 1 - tr_reached['r']
                start = len(traj_pos) - tr_reached['f']
                # end = len(traj_pos) - 1 - tr_reached['f']
                end = len(traj_pos) - tr_reached['r']
                # step = -1
                step = 1
                threshold = 0.2
                print(hand, "rev:", [i for i in range(start, end, step)])
            trajs_reached = []
            for i in range(start, end, step):
                if not no_punch_mode:
                    traj_pos[i] = traj_pos[i - 1] + pos_step_global
                    if np.linalg.norm(desired_punch_target - traj_pos[i]) < threshold:
                        trajs_reached.append(i)
                        if fwd_motion:
                            tr_reached['f'] += 1
                        elif rev_motion:
                            tr_reached['r'] += 1
                            # if np.linalg.norm(desired_punch_target - traj_pos[tr_mid_idx]) < 0.06:
                            if tr_reached['r'] == tr_mid_idx - 1:
                                wrist_reached = True

                elif no_punch_mode:
                    traj_pos[i] = traj_pos[tr_mid_idx]

            if not no_punch_mode:
                print(hand, tr_reached, trajs_reached)

            return traj_pos, tr_reached, wrist_reached

        def _tr_update_vel(traj_vels_blend, tr_mid_idx, traj_reached, fwd_motion=True):

            if fwd_motion:
                start = tr_mid_idx + 1
                end = len(traj_vels_blend) - traj_reached['f']
                step = 1
            else:
                start = len(traj_vels_blend) - traj_reached['f']
                end = len(traj_vels_blend)
                # step = -1
                step = 1

            for i in range(start, end, step):
                scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)
                # scale_pos = 0.4
                # iterates over predictions
                if not no_punch_mode:
                    traj_vels_blend[i] = utils.glm_mix(traj_vels_blend[i],
                                                       traj_pos_blend[i] - traj_pos_blend[i - 1], scale_pos)
                elif no_punch_mode:
                    traj_vels_blend[i] = np.array([0, 0, 0])

            return traj_vels_blend

        for hand in ['left', 'right']:
            # for hand in ['left']:
            no_punch_mode = False

            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_pos, dtype=np.float64)
                wrist_lp = left_wr_lp
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                desired_punch_target = desired_left_punch_target
                # TODO get average local desired_punch_target_reverse from dataset
                desired_punch_target_reverse = _loc_to_glob(left_shoulder_pos + np.array([0, 0, 0.2]))
                # TODO get global magnitude of
                wrist_pos_avg_diff = self.left_wrist_pos_avg_diff
                traj_reached = self.traj_reached_left_wrist
                wrist_reached = self.wrist_reached_left_wrist
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_pos, dtype=np.float64)
                wrist_lp = right_wr_lp
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                desired_punch_target = desired_right_punch_target
                desired_punch_target_reverse = _loc_to_glob(right_shoulder_pos + np.array([0, 0, 0.23]))
                wrist_pos_avg_diff = self.right_wrist_pos_avg_diff
                traj_reached = self.traj_reached_right_wrist
                wrist_reached = self.wrist_reached_right_wrist

            no_punch_target = np.array([0, 0, 0])
            if np.sum(desired_punch_target) == 0:
                desired_punch_target = np.array(no_punch_target)
                no_punch_mode = True
            else:
                desired_punch_target = _loc_to_glob(
                    _glob_to_loc(desired_punch_target, root_position, root_rotation) + np.array([0, 0, 0.1]))
            tr_mid_idx = self.median_idx
            wrist_pos_avg_diff_g = _loc_to_glob(wrist_pos_avg_diff, True)
            wrist_pos_avg_diff_g = 0.02
            # wrist_pos_avg_diff_g = 0.016594728784002494
            # wrist_pos_avg_diff_g = wrist_pos_avg_diff
            # wrist_gp = _loc_to_glob(wrist_lp)
            wrist_gp = traj_pos_blend[tr_mid_idx]

            # if traj_reached['f'] >= tr_mid_idx - 1 and traj_reached['r'] >= tr_mid_idx - 1 \
            #         and self.punch_frames > 0:
            if traj_reached['f'] >= tr_mid_idx - 1 and wrist_reached is True and self.punch_frames > 0:
                traj_reached['f'] = 0
                traj_reached['r'] = 0
                # self.wrist_reached_left_wrist = False
                print("----------------", self.punch_frames)
                self.punch_frames = 0
                desired_punch_target == np.array([0, 0, 0])
                wrist_reached = False

            if 0 < traj_reached['f'] < tr_mid_idx - 1:
                fwd = True
                rev = True
            elif traj_reached['f'] == 0:
                fwd = True
                rev = False
            else:
                traj_reached_updated = traj_reached.copy()
                fwd = False
                rev = True

            if not no_punch_mode:
                self.punch_frames += 1

            if fwd:
                pos_step_g = utils.normalize(desired_punch_target - wrist_gp) * wrist_pos_avg_diff_g
                # pos_step_g = (desired_punch_target - wrist_gp) / post_step_div_factor
                traj_pos_blend, traj_reached_updated, wrist_reached = _tr_update_pos_g(traj_pos_blend, tr_mid_idx,
                                                                                       desired_punch_target,
                                                                                       no_punch_mode, traj_reached,
                                                                                       wrist_reached,
                                                                                       pos_step_g, hand, fwd_motion=fwd,
                                                                                       rev_motion=False)

                traj_vels_blend = _tr_update_vel(traj_vels_blend, tr_mid_idx, traj_reached, fwd_motion=fwd)

            if rev:
                pos_step_rev_g = utils.normalize(desired_punch_target_reverse - wrist_gp) * wrist_pos_avg_diff_g
                # pos_step_rev_g = (desired_punch_target_reverse - wrist_gp) / post_step_div_factor
                traj_pos_blend, traj_reached_updated_rev, wrist_reached = _tr_update_pos_g(traj_pos_blend, tr_mid_idx,
                                                                                           desired_punch_target_reverse,
                                                                                           no_punch_mode, traj_reached,
                                                                                           wrist_reached,
                                                                                           pos_step_rev_g, hand,
                                                                                           fwd_motion=False,
                                                                                           rev_motion=rev)
                traj_vels_blend = _tr_update_vel(traj_vels_blend, tr_mid_idx, traj_reached, fwd_motion=False)

                traj_reached_updated['r'] = traj_reached_updated_rev['r']

            self._update_wrist_traj(traj_pos_blend, traj_vels_blend, traj_reached_updated, wrist_reached, hand)

    def step_forward(self, pred_root_vel, pred_fwd_dir, pred_local_wrist_vels, pred_local_wrist_pos, pred_root_pos,
                     curr_punch_labels):
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

        def _curr_frame_update(local_vel, xz_to_x0yz=False):
            local_vel = np.array(local_vel)
            if xz_to_x0yz:
                local_vel = utils.xz_to_x0yz(local_vel)
            tr_update = local_vel.reshape(1, self.n_dims)

            return tr_update

        def _convert_wrist_lp_to_gp(wrist_pos, arm):
            wrist_pos = wrist_pos.reshape(1, self.n_dims)
            return self.convert_local_to_global(wrist_pos, arg_type="pos", arm=arm)

        idx = self.median_idx
        root_lp = pred_root_pos
        root_gp = _convert_wrist_lp_to_gp(root_lp, arm=None)
        root_gp[0][1] = 0

        root_tr_update = _curr_frame_update(pred_root_vel, xz_to_x0yz=True)
        self.traj_root_pos[idx] = root_gp
        self.traj_root_pos[idx] = self.traj_root_pos[idx] + root_tr_update
        self.traj_root_vels[idx] = utils.glm_mix(self.traj_root_vels[idx], root_tr_update, 0.9)

        pred_fwd_dir = pred_fwd_dir.ravel()
        pred_fwd_dir_x, pred_fwd_dir_z = pred_fwd_dir[0], pred_fwd_dir[1]
        rotational_vel = math.atan2(pred_fwd_dir_x, pred_fwd_dir_z)
        self.traj_root_directions[idx] = utils.rot_around_z_3d(self.traj_root_directions[idx],
                                                               rotational_vel)
        self.traj_root_rotations[idx] = utils.z_angle(self.traj_root_directions[idx])

        right_wr_lp, left_wr_lp = pred_local_wrist_pos
        right_wr_gp = _convert_wrist_lp_to_gp(right_wr_lp, arm="right")
        left_wr_gp = _convert_wrist_lp_to_gp(left_wr_lp, arm="left")
        right_wr_v, left_wr_v = pred_local_wrist_vels
        right_tr_update = _curr_frame_update(right_wr_v)
        right_tr_update = self.convert_local_to_global(right_tr_update, arg_type='vels')
        if curr_punch_labels['right'] == 0:
            right_tr_update = np.array([0, 0, 0])
        self.traj_right_wrist_pos[idx] = right_wr_gp
        if curr_punch_labels['right'] != 0:
            self.traj_right_wrist_pos[idx] = self.traj_right_wrist_pos[idx] + right_tr_update
        self.traj_right_wrist_vels[idx] = right_tr_update

        left_tr_update = _curr_frame_update(left_wr_v)
        left_tr_update = self.convert_local_to_global(left_tr_update, arg_type='vels')
        if curr_punch_labels['left'] == 0:
            left_tr_update = np.array([0, 0, 0])
        self.traj_left_wrist_pos[idx] = left_wr_gp
        if curr_punch_labels['left'] != 0:
            self.traj_left_wrist_pos[idx] = self.traj_left_wrist_pos[idx] + left_tr_update
        self.traj_left_wrist_vels[idx] = left_tr_update

    def update_from_predict(self, prediction, curr_punch_labels):
        """
        Update trajectory from prediction.
        Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y.

        Arguments:
            prediction {np.array(4 * (6 * 2))} -- vector containing the network output regarding future trajectory positions and directions
        """
        # These predictions from the neural network are only for the trajectory points considered for the neural network
        prediction = list(prediction)

        right_p_lab = curr_punch_labels['right']
        left_p_lab = curr_punch_labels['left']

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
        # TODO Check whether to apply wrist preds or not to avoid network generated punching when user is not
        #  punching ==> Did not work
        pred_rp_tr, pred_rv_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr = prediction

        half_pred_window = self.median_idx // self.traj_step
        pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(half_pred_window, self.n_dims - 1))
        self.traj_root_pos[self.median_idx + 1:] = self.convert_local_to_global(
            utils.xz_to_x0yz(pred_rp_tr,
                             axis=1))  # TODO maybe apply some correction to prevent drifting ==> suppress movement smaller than a certain amount
        pred_rv_tr = _smooth_predictions(pred_rv_tr.reshape(half_pred_window, self.n_dims - 1))
        self.traj_root_vels[self.median_idx + 1:] = self.convert_local_to_global(
            utils.xz_to_x0yz(pred_rv_tr, axis=1),
            arg_type='vels')

        # pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(self.n_frames_tr_win // self.traj_step, self.n_dims - 1))
        # self.traj_root_directions[self.median_idx + 1:] = self.convert_local_to_global(
        #     utils.normalize(utils.xz_to_x0yz(pred_fwd_dir)), arg_type="dir")
        # pred_r_rot = np.array(
        #     [utils.z_angle(self.traj_directions[i]) for i in range(self.median_idx + 1, self.n_frames_tr_win)])
        # self.traj_root_rotations[self.median_idx + 1:] = pred_r_rot

        # if curr_punch_labels['right'] != 0:
        #     start = np.array(self.right_wrist_pos_no_punch_dist["start"])
        #     end = np.array(self.right_wrist_pos_no_punch_dist["end"])
        #     step = (start - end) / (self.n_frames_tr_win * .5)
        #     pred_rwp_tr = np.array([start + step * i for i in range(int(self.n_frames_tr_win * .5))])
        # if curr_punch_labels['left'] != 0:
        #     start = np.array(self.left_wrist_pos_no_punch_dist["start"])
        #     end = np.array(self.left_wrist_pos_no_punch_dist["end"])
        #     step = (start - end) / (self.n_frames_tr_win * .5)
        #     pred_lwp_tr = np.array([start + step * i for i in range(int(self.n_frames_tr_win * .5))])

        pred_rwp_tr = _smooth_predictions(pred_rwp_tr.reshape(half_pred_window, self.n_dims))
        self.traj_right_wrist_pos[self.median_idx + 1:] = self.convert_local_to_global(pred_rwp_tr, arg_type='pos',
                                                                                       arm="right")

        pred_rwv_tr = _smooth_predictions(pred_rwv_tr.reshape(half_pred_window, self.n_dims))
        self.traj_right_wrist_vels[self.median_idx + 1:] = self.convert_local_to_global(pred_rwv_tr, arg_type='vels',
                                                                                        arm="right")

        pred_lwp_tr = _smooth_predictions(pred_lwp_tr.reshape(half_pred_window, self.n_dims))
        self.traj_left_wrist_pos[self.median_idx + 1:] = self.convert_local_to_global(pred_lwp_tr, arg_type='pos',
                                                                                      arm="left")
        pred_lwv_tr = _smooth_predictions(pred_lwv_tr.reshape(half_pred_window, self.n_dims))
        self.traj_left_wrist_vels[self.median_idx + 1:] = self.convert_local_to_global(pred_lwv_tr, arg_type='vels',
                                                                                       arm="left")

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
            curr_point = utils.rot_around_z_3d(curr_point, root_rot_copy, inverse=True)
            arr_copy[i] = curr_point

        for i in range(len(arr_copy)):
            curr_point = arr_copy[i]
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

    def reset(self, right_shoulder_lp, left_shoulder_lp, start_location=[0.0, 0.0, 0.0], start_orientation=[0.0],
              start_direction=[0, 0, 1]):
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
        # TODO control rotations using these dirs
        self.traj_root_directions = np.array([start_direction] * self.n_frames_tr_win)

        ### Trajectory info of the hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)

        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)

        self.traj_reached_left_wrist = {'f': 0, 'r': 0}
        self.traj_reached_right_wrist = {'f': 0, 'r': 0}

        self.wrist_reached_left_wrist = False
        self.wrist_reached_right_wrist = False

        # for idx in range(self.median_idx + 1):
        #     self.traj_root_pos[idx] = np.array(start_location)
        #
        #     self.traj_root_rotations[idx] = np.array(start_orientation)
        #     self.traj_root_directions[idx] = np.array(start_direction)
        #
        # right_shoulder_lp = np.array(right_shoulder_lp).reshape(1, len(right_shoulder_lp))
        # right_shoulder_gp = self.convert_local_to_global(right_shoulder_lp, arg_type='pos')
        #
        # left_shoulder_lp = np.array(left_shoulder_lp).reshape(1, len(left_shoulder_lp))
        # left_shoulder_gp = self.convert_local_to_global(left_shoulder_lp, arg_type='pos')
        #
        # for idx in range(self.median_idx + 1):
        #     self.traj_left_wrist_pos[idx] = left_shoulder_gp.ravel()
        #     self.traj_right_wrist_pos[idx] = right_shoulder_gp.ravel()
