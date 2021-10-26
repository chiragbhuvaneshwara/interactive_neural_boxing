"""author: Chirag Bhuvaneshwara """
import numpy as np
import math

from vis.backend.controller import utils


class Trajectory:
    """
    This class contains data and functionality for trajectory control.
    All trajectory data inside this class should be maintained in global space.
    """

    def __init__(self, data_configuration):
        self.bone_map = data_configuration["bone_map"]
        self.n_tr_samples_root = data_configuration['num_traj_samples_root']  # 10
        self.n_tr_samples_wrist = data_configuration['num_traj_samples_wrist']  # 10
        # TODO: (DONE) Replace in below code self.traj_step_root and self.traj_step_wrist to frame rate div
        # self.traj_step_root = data_configuration['traj_step_root']  # 5
        # self.traj_step_wrist = data_configuration['traj_step_wrist']  # 5
        self.traj_step_root = data_configuration['frame_rate_div']  # 5
        self.traj_step_wrist = data_configuration['frame_rate_div']  # 5
        # self.left_wrist_pos_avg_diff = np.array(data_configuration['left_wrist_pos_avg_diff']).ravel()
        # self.right_wrist_pos_avg_diff = np.array(data_configuration['right_wrist_pos_avg_diff']).ravel()
        # self.left_wrist_pos_avg_diff_gp = np.array(data_configuration['left_wrist_pos_avg_diff']).ravel()
        # self.right_wrist_pos_avg_diff_gp = np.array(data_configuration['right_wrist_pos_avg_diff']).ravel()
        # self.left_wrist_pos_no_punch_dist = data_configuration['left_wrist_no_punch']
        # self.right_wrist_pos_no_punch_dist = data_configuration['right_wrist_no_punch']

        # 10 * 5 = 50fps trajectory window
        self.n_frames_tr_win_root = self.n_tr_samples_root * self.traj_step_root
        self.n_frames_tr_win_wrist = self.n_tr_samples_wrist * self.traj_step_wrist
        self.median_idx_root = self.n_frames_tr_win_root // 2
        self.median_idx_wrist = self.n_frames_tr_win_wrist // 2

        self.n_dims = 3

        ### Trajectory info of the root
        self.traj_root_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_root)
        self.traj_root_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_root)
        self.traj_root_rotations = np.zeros(self.n_frames_tr_win_root)
        self.traj_root_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames_tr_win_root)

        ### Trajectory info of the right hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)
        # self.traj_right_punch_labels = np.array([0] * self.n_frames_tr_win)

        ### Trajectory info of the left hand
        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)
        # self.traj_left_punch_labels = np.array([0] * self.n_frames_tr_win)

        self.traj_reached_left_wrist = {'f': 0, 'r': 0}
        self.traj_reached_right_wrist = {'f': 0, 'r': 0}

        self.punch_completed_left = False
        self.punch_completed_right = False

        self.punch_half_completed_left = False
        self.punch_half_completed_right = False

        self.foot_drifting = np.zeros(3)
        # self.blend_bias = 2.0
        self.blend_bias = 1.5  # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
        # self.blend_bias = 10  # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
        self.punch_frames_left = 0
        self.punch_frames_right = 0
        self.punch_frames_left_half_comp = 0
        self.punch_frames_right_half_comp = 0

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
        tr_root_dirs_local = self.convert_global_to_local(self.traj_root_directions, root_position, root_rotation,
                                                          arg_type='dirs')
        tr_right_pos_local = self.convert_global_to_local(self.traj_right_wrist_pos, root_position, root_rotation,
                                                          arg_type='pos', arm='right')
        tr_left_pos_local = self.convert_global_to_local(self.traj_left_wrist_pos, root_position, root_rotation,
                                                         arg_type='pos', arm='left')
        tr_right_vels_local = self.convert_global_to_local(self.traj_right_wrist_vels, root_position, root_rotation,
                                                           arg_type='vels', arm='right')
        tr_left_vels_local = self.convert_global_to_local(self.traj_left_wrist_vels, root_position, root_rotation,
                                                          arg_type='vels', arm='left')

        # Deleting Y axis since root pos doesnt contain Y axis
        input_root_pos = np.delete(tr_root_pos_local[::self.traj_step_root], obj=1, axis=1).ravel()
        input_root_vels = np.delete(tr_root_vels_local[::self.traj_step_root], obj=1, axis=1).ravel()
        input_root_dirs = np.delete(tr_root_dirs_local[::self.traj_step_root], obj=1, axis=1).ravel()
        input_right_wrist_pos = tr_right_pos_local[::self.traj_step_wrist].ravel()
        input_left_wrist_pos = tr_left_pos_local[::self.traj_step_wrist].ravel()
        input_right_wrist_vels = tr_right_vels_local[::self.traj_step_wrist].ravel()
        input_left_wrist_vels = tr_left_vels_local[::self.traj_step_wrist].ravel()

        return input_root_pos, input_root_vels, input_root_dirs, \
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

        trajectory_positions_blend = np.array(self.traj_root_pos)
        tr_mid_idx = self.median_idx_root
        for i in range(tr_mid_idx + 1, len(trajectory_positions_blend)):
            # adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)

            # TODO : Finalize which bias value to use
            # scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)
            scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), 0.5)

            # ith pos = i-1 th pos + combo of curr velocity and user requested velocity
            trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + \
                                            utils.glm_mix(self.traj_root_pos[i] - self.traj_root_pos[i - 1], target_vel,
                                                          scale_pos)

            self.traj_root_vels[i] = utils.glm_mix(self.traj_root_vels[i],
                                                   target_vel, scale_pos)
            scale_dir = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), 0.1)
            # self.traj_root_directions[i] = utils.mix_directions(self.traj_root_directions[i], target_dir,
            #                                                     scale_dir)
            self.traj_root_directions[i] = self.traj_root_directions[i]

        for i in range(tr_mid_idx + 1, len(trajectory_positions_blend)):
            self.traj_root_pos[i] = trajectory_positions_blend[i]

        # compute trajectory rotations
        # for i in range(0, self.n_frames_tr_win_root):
        for i in range(tr_mid_idx + 1, self.n_frames_tr_win_root):
            self.traj_root_rotations[i] = utils.z_angle(self.traj_root_directions[i])

        # tr_root_rots_degrees = [i * 180 / math.pi for i in self.traj_root_rotations]
        # print(self.traj_root_directions)

    def _update_wrist_traj(self, traj_pos_blend, traj_vels_blend, punch_frames, p_frames_half_comp, traj_reached,
                           punch_half_done,
                           punch_done, hand):
        if hand == 'left':
            self.traj_left_wrist_pos = traj_pos_blend[:]
            self.traj_left_wrist_vels = traj_vels_blend[:]
            self.punch_completed_left = punch_done
            self.punch_half_completed_left = punch_half_done
            self.traj_reached_left_wrist = traj_reached
            self.punch_frames_left_half_comp = p_frames_half_comp
            self.punch_frames_left = punch_frames

        elif hand == 'right':
            self.traj_right_wrist_pos = traj_pos_blend[:]
            self.traj_right_wrist_vels = traj_vels_blend[:]
            self.punch_completed_right = punch_done
            self.punch_half_completed_right = punch_half_done
            self.traj_reached_right_wrist = traj_reached
            self.punch_frames_right_half_comp = p_frames_half_comp
            self.punch_frames_right = punch_frames

    def compute_future_wrist_trajectory(self, desired_right_punch_target, desired_left_punch_target,
                                        right_shoulder_pos, left_shoulder_pos, right_wr_lp, left_wr_lp, ):
        """
        Performs blending of the future trajectory for predicted trajectory info passed in.
        :param desired_right_punch_target: np.array(3), local space
        :param desired_left_punch_target: np.array(3), local space
        :param desired_left_punch_label: np.array(1)
        :param desired_right_punch_label: np.array(1)
        :param right_shoulder_pos:
        :param left_shoulder_pos:
        @param left_wr_lp:
        @param right_wr_lp:
        :return:

        """

        def _loc_to_glob(lp):
            pos = np.array(lp)
            pos = pos.reshape(1, len(pos))
            pos_g = self.convert_local_to_global(pos, arg_type='pos')

            return pos_g.ravel()

        def _glob_to_loc(gp, rp, rr):
            pos = np.array(gp)
            pos = pos.reshape(1, len(pos))
            pos_g = self.convert_global_to_local(pos, rp, rr, arg_type='pos')
            return pos_g.ravel()

        def _tr_update_pos_g(wrist_glob_p, traj_pos, tr_mid_idx, desired_punch_target, no_punch_mode,
                             tr_reached, punch_completed, pos_step_global, hand, fwd_motion=True, rev_motion=False):

            punch_completed_status = punch_completed
            tr_reached = tr_reached.copy()
            if fwd_motion:
                start = tr_mid_idx + 1
                end = len(traj_pos)
                step = 1
                threshold = 0.09
            if rev_motion:
                start = len(traj_pos) - tr_reached['f']
                end = len(traj_pos)
                step = 1
                threshold = 0.15
            trajs_reached = []
            count = 1
            for i in range(start, end, step):
                scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)
                if not no_punch_mode:
                    traj_pos[i] = wrist_glob_p + (pos_step_global * count)
                    if np.linalg.norm(desired_punch_target - traj_pos[tr_mid_idx]) < threshold:
                        if fwd_motion:
                            tr_reached['f'] = tr_mid_idx - 1
                        elif rev_motion:
                            tr_reached['r'] = tr_mid_idx - 1
                            if tr_reached['r'] == tr_mid_idx - 1:
                                punch_completed_status = True

                    count += 1
                elif no_punch_mode:
                    traj_pos[i] = traj_pos[tr_mid_idx]

            return traj_pos, tr_reached, punch_completed_status

        def _tr_update_vel(traj_vels_blend, tr_mid_idx, traj_reached, fwd_motion=True):

            if fwd_motion:
                start = tr_mid_idx + 1
                end = len(traj_vels_blend) - traj_reached['f']
                step = 1
            else:
                start = len(traj_vels_blend) - traj_reached['f']
                end = len(traj_vels_blend)
                step = 1

            for i in range(start, end, step):
                scale_pos = 1.0 - pow(1.0 - (i - tr_mid_idx) / (1.0 * tr_mid_idx), self.blend_bias)
                # iterates over predictions
                if not no_punch_mode:
                    traj_vels_blend[i] = utils.glm_mix(traj_vels_blend[i],
                                                       (traj_pos_blend[i] - traj_pos_blend[
                                                           i - 1]) / self.traj_step_wrist, scale_pos)
                elif no_punch_mode:
                    traj_vels_blend[i] = np.array([0, 0, 0])

            return traj_vels_blend

        for hand in ['left', 'right']:
            no_punch_mode = False

            if hand == 'left':
                traj_pos_blend = np.array(self.traj_left_wrist_pos, dtype=np.float64)
                wrist_gp = _loc_to_glob(left_wr_lp)
                traj_vels_blend = np.array(self.traj_left_wrist_vels, dtype=np.float64)
                desired_punch_target = desired_left_punch_target
                # TODO get average local desired_punch_target_reverse from dataset
                desired_punch_target_reverse = _loc_to_glob(left_shoulder_pos + np.array([0, 0, 0.18]))
                # desired_punch_target_reverse = _loc_to_glob(left_shoulder_pos)
                traj_reached = self.traj_reached_left_wrist
                punch_completed = self.punch_completed_left
                punch_half_completed = self.punch_half_completed_left
                punch_frames = self.punch_frames_left
                punch_frames_half_comp = self.punch_frames_left_half_comp
            elif hand == 'right':
                traj_pos_blend = np.array(self.traj_right_wrist_pos, dtype=np.float64)
                wrist_gp = _loc_to_glob(right_wr_lp)
                traj_vels_blend = np.array(self.traj_right_wrist_vels, dtype=np.float64)
                desired_punch_target = desired_right_punch_target
                desired_punch_target_reverse = _loc_to_glob(right_shoulder_pos + np.array([0, 0, 0.21]))
                # desired_punch_target_reverse = _loc_to_glob(right_shoulder_pos)
                traj_reached = self.traj_reached_right_wrist
                punch_completed = self.punch_completed_right
                punch_half_completed = self.punch_half_completed_right
                punch_frames = self.punch_frames_right
                punch_frames_half_comp = self.punch_frames_right_half_comp

            tr_mid_idx = self.median_idx_wrist
            wrist_gp = traj_pos_blend[tr_mid_idx]

            # wrist_pos_avg_diff_g = 0.04
            wrist_pos_avg_diff_g = 0.06
            # wrist_pos_avg_diff_g = 0.08

            # Reset after punch completed
            if traj_reached['f'] >= tr_mid_idx - 1 and punch_completed is True and punch_frames > 0:
                traj_reached['f'] = 0
                traj_reached['r'] = 0
                punch_frames_half_comp = 0
                punch_frames = 0
                punch_completed = False

            # Condition to force half punch completed if punch target is not reachable
            if punch_frames >= 25 and not punch_half_completed:
                traj_reached['f'] = tr_mid_idx - 1

            # Condition to force punch completed if shoulder target is not reachable
            if punch_frames >= 50:
                no_punch_mode = True
                traj_reached['f'] = tr_mid_idx - 1
                traj_reached['r'] = tr_mid_idx - 1
                punch_completed = True

            # Checking if punch mode or not
            if np.sum(desired_punch_target) == 0:
                no_punch_mode = True
            else:
                punch_frames += 1

            # Setting up forward and backward punch modes
            if 0 < traj_reached['f'] < tr_mid_idx - 1:
                fwd = True
                rev = False
            elif traj_reached['f'] == 0:
                fwd = True
                rev = False
            else:
                traj_reached_updated = traj_reached.copy()
                fwd = False
                rev = True

            # TODO this method should apply the interpolation to all the points maintained in the trajectory class
            # Compute motion towards punch target
            if fwd:
                # pos_step_g = utils.normalize(desired_punch_target - wrist_gp) * wrist_pos_avg_diff_g
                pos_step_g = (desired_punch_target - wrist_gp) * wrist_pos_avg_diff_g
                traj_pos_blend, traj_reached_updated, punch_completed = _tr_update_pos_g(wrist_gp, traj_pos_blend,
                                                                                         tr_mid_idx,
                                                                                         desired_punch_target,
                                                                                         no_punch_mode, traj_reached,
                                                                                         punch_completed,
                                                                                         pos_step_g, hand,
                                                                                         fwd_motion=fwd,
                                                                                         rev_motion=False)

                traj_vels_blend = _tr_update_vel(traj_vels_blend, tr_mid_idx, traj_reached, fwd_motion=fwd)

            # Compute motion towards shoulder target
            if rev:
                # pos_step_rev_g = utils.normalize(desired_punch_target_reverse - wrist_gp) * wrist_pos_avg_diff_g
                pos_step_rev_g = (desired_punch_target_reverse - wrist_gp) * wrist_pos_avg_diff_g
                traj_pos_blend, traj_reached_updated_rev, punch_completed = _tr_update_pos_g(wrist_gp, traj_pos_blend,
                                                                                             tr_mid_idx,
                                                                                             desired_punch_target_reverse,
                                                                                             no_punch_mode,
                                                                                             traj_reached,
                                                                                             punch_completed,
                                                                                             pos_step_rev_g, hand,
                                                                                             fwd_motion=False,
                                                                                             rev_motion=rev)
                traj_vels_blend = _tr_update_vel(traj_vels_blend, tr_mid_idx, traj_reached, fwd_motion=False)

                traj_reached_updated['r'] = traj_reached_updated_rev['r']

            if ((traj_reached_updated["f"] == tr_mid_idx - 1) and (traj_reached_updated[
                                                                       "r"] == 0)) and punch_frames_half_comp == 0:
                punch_half_completed = True
                punch_frames_half_comp = punch_frames

            else:
                punch_half_completed = False

            if punch_completed:
                print("Punch details (half, full) :", punch_frames_half_comp - 1, punch_frames - 1)

            self._update_wrist_traj(traj_pos_blend, traj_vels_blend, punch_frames, punch_frames_half_comp,
                                    traj_reached_updated,
                                    punch_half_completed, punch_completed, hand)

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

        tr_mid_idx_root = self.median_idx_root
        tr_mid_idx_wrist = self.median_idx_wrist

        ## current trajectory
        self.traj_root_pos[:tr_mid_idx_root] = self.traj_root_pos[1:tr_mid_idx_root + 1]
        self.traj_root_vels[:tr_mid_idx_root] = self.traj_root_vels[1:tr_mid_idx_root + 1]
        self.traj_root_directions[:tr_mid_idx_root] = self.traj_root_directions[1:tr_mid_idx_root + 1]
        self.traj_root_rotations[:tr_mid_idx_root] = self.traj_root_rotations[1:tr_mid_idx_root + 1]

        self.traj_right_wrist_pos[:tr_mid_idx_wrist] = self.traj_right_wrist_pos[1:tr_mid_idx_wrist + 1]
        self.traj_left_wrist_pos[:tr_mid_idx_wrist] = self.traj_left_wrist_pos[1:tr_mid_idx_wrist + 1]
        self.traj_right_wrist_vels[:tr_mid_idx_wrist] = self.traj_right_wrist_vels[1:tr_mid_idx_wrist + 1]
        self.traj_left_wrist_vels[:tr_mid_idx_wrist] = self.traj_left_wrist_vels[1:tr_mid_idx_wrist + 1]

        def _curr_frame_update(local_vel, xz_to_x0yz=False):
            local_vel = np.array(local_vel)
            if xz_to_x0yz:
                local_vel = utils.xz_to_x0yz(local_vel)
            tr_update = local_vel.reshape(1, self.n_dims)

            return tr_update

        def _convert_wrist_lp_to_gp(wrist_pos, arm):
            wrist_pos = wrist_pos.reshape(1, self.n_dims)
            return self.convert_local_to_global(wrist_pos, arg_type="pos", arm=arm)

        idx = self.median_idx_root
        root_lp = pred_root_pos
        root_gp = _convert_wrist_lp_to_gp(root_lp, arm=None)
        root_gp[0][1] = 0

        root_tr_update = _curr_frame_update(pred_root_vel, xz_to_x0yz=True)
        self.traj_root_pos[idx] = root_gp
        self.traj_root_pos[idx] = self.traj_root_pos[idx] + root_tr_update
        # TODO: Needed to disable foot_drifting addition to enable better stepping
        # self.traj_root_pos[idx] = self.traj_root_pos[idx] + root_tr_update + \
        #                           self.foot_drifting.reshape(1, len(self.foot_drifting))
        self.traj_root_vels[idx] = utils.glm_mix(self.traj_root_vels[idx], root_tr_update, 0.9)

        pred_fwd_dir = pred_fwd_dir.ravel()
        pred_fwd_dir_x, pred_fwd_dir_z = pred_fwd_dir[0], pred_fwd_dir[1]
        rotational_vel = math.atan2(pred_fwd_dir_x, pred_fwd_dir_z)

        # TODO enable or disable rotations
        self.traj_root_directions[idx] = utils.rot_around_z_3d(self.traj_root_directions[idx],
                                                               0)
        # rotational_vel)

        self.traj_root_rotations[idx] = utils.z_angle(self.traj_root_directions[idx])

        idx = self.median_idx_wrist
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

    def update_from_predict(self, prediction):
        """
        Update trajectory from prediction.
        Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y.

        Arguments:
            prediction {np.array(4 * (6 * 2))} -- vector containing the network output regarding future trajectory positions and directions
        """
        prediction = list(prediction)

        def _smooth_predictions(pred_arr, root=True):
            if root:
                tr_step = self.traj_step_root
                n_frames = self.n_frames_tr_win_root
                n_tr_samples = self.n_tr_samples_root
                half_pred_window = self.median_idx_root // tr_step
                tr_mid_idx = self.median_idx_root
            else:
                tr_step = self.traj_step_wrist
                n_frames = self.n_frames_tr_win_wrist
                n_tr_samples = self.n_tr_samples_wrist
                half_pred_window = self.median_idx_wrist // tr_step
                tr_mid_idx = self.median_idx_wrist

            combo_idxs_1 = []
            combo_idxs_2 = []
            weights = []
            for i in range(tr_mid_idx + 1, n_frames):
                weight = ((i - n_frames / 2) / tr_step) % 1.0
                weights.append(weight)
                combo_idxs_1.append(i // tr_step - half_pred_window)
                combo_idxs_2.append((i // tr_step) + (
                    1 if i < (n_frames - n_tr_samples) else 0) - half_pred_window)

            # TODO Fix above loop and no need for below postprocessing fix
            combo_idxs_2 = [i if i <= combo_idxs_2[-1] else combo_idxs_2[-1] for i in combo_idxs_2]

            weights = np.array(weights).reshape(len(weights), 1)
            a = np.array([pred_arr[i] for i in combo_idxs_1])
            b = np.array([pred_arr[i] for i in combo_idxs_2])
            part_a = np.multiply((1 - weights), a)
            part_b = np.multiply(weights, b)
            res = part_a + part_b

            return res

        def _convert_dir_lp_to_gp(root_dirs):
            root_dirs = root_dirs.reshape(1, self.n_dims)
            return self.convert_local_to_global(root_dirs, arg_type="dir")

        # The predictions after applying _smooth_predictions will be producing entries for all trajectories
        # maintained in trajectory class
        pred_rp_tr, pred_rv_tr, pred_rdir_tr, pred_rwp_tr, pred_lwp_tr, pred_rwv_tr, pred_lwv_tr = prediction

        median_idx_root = self.median_idx_root
        half_pred_window_root = median_idx_root // self.traj_step_root

        pred_rp_tr = _smooth_predictions(pred_rp_tr.reshape(half_pred_window_root, self.n_dims - 1))
        self.traj_root_pos[median_idx_root + 1:] = self.convert_local_to_global(
            utils.xz_to_x0yz(pred_rp_tr, axis=1))
        pred_rv_tr = _smooth_predictions(pred_rv_tr.reshape(half_pred_window_root, self.n_dims - 1))
        self.traj_root_vels[median_idx_root + 1:] = self.convert_local_to_global(
            utils.xz_to_x0yz(pred_rv_tr, axis=1), arg_type='vels')

        pred_rdir_tr = _smooth_predictions(pred_rdir_tr.reshape(half_pred_window_root, self.n_dims - 1))
        pred_rdir_tr = np.array([utils.normalize(utils.xz_to_x0yz(i)) for i in pred_rdir_tr])

        # self.traj_root_directions[self.median_idx_root + 1:] = self.convert_local_to_global(
        #     pred_rdir_tr, arg_type="dir")
        #
        # pred_r_rot = np.array(
        #     [utils.z_angle(self.traj_root_directions[i]) for i in
        #      range(self.median_idx_root + 1, self.n_frames_tr_win_root)])
        # self.traj_root_rotations[self.median_idx_root + 1:] = pred_r_rot

        median_idx_wrist = self.median_idx_wrist
        half_pred_window_wrist = self.median_idx_wrist // self.traj_step_wrist

        pred_rwp_tr = _smooth_predictions(pred_rwp_tr.reshape(half_pred_window_wrist, self.n_dims), root=False)
        self.traj_right_wrist_pos[median_idx_wrist + 1:] = self.convert_local_to_global(pred_rwp_tr, arg_type='pos',
                                                                                        arm="right")

        pred_rwv_tr = _smooth_predictions(pred_rwv_tr.reshape(half_pred_window_wrist, self.n_dims), root=False)
        self.traj_right_wrist_vels[median_idx_wrist + 1:] = self.convert_local_to_global(pred_rwv_tr,
                                                                                         arg_type='vels',
                                                                                         arm="right")

        pred_lwp_tr = _smooth_predictions(pred_lwp_tr.reshape(half_pred_window_wrist, self.n_dims), root=False)
        self.traj_left_wrist_pos[median_idx_wrist + 1:] = self.convert_local_to_global(pred_lwp_tr, arg_type='pos',
                                                                                       arm="left")
        pred_lwv_tr = _smooth_predictions(pred_lwv_tr.reshape(half_pred_window_wrist, self.n_dims), root=False)
        self.traj_left_wrist_vels[median_idx_wrist + 1:] = self.convert_local_to_global(pred_lwv_tr,
                                                                                        arg_type='vels',
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
        root_pos = np.array(self.traj_root_pos[self.median_idx_root])
        root_rot = np.array(self.traj_root_rotations[self.median_idx_root])
        if arm == "left" or arm == "right":
            root_pos[1] = 0

        for i in range(len(arr_copy)):
            arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], root_rot)

            if arg_type == 'pos':
                arr_copy[i] = arr_copy[i] + root_pos

        return arr_copy

    def get_world_pos_rot(self):
        pos = np.array(self.traj_root_pos[self.median_idx_root])
        pos[1] = 0.0

        rot = self.traj_root_rotations[self.median_idx_root]
        return pos, rot

    def get_previous_pos_rot(self):
        pos = np.array(self.traj_root_pos[self.median_idx_root - 1])
        pos[1] = 0.0

        rot = self.traj_root_rotations[self.median_idx_root - 1]
        return pos, rot

    def reset(self, start_location=[0.0, 0.0, 0.0], start_orientation=[0.0],
              start_direction=[0, 0, 1]):
        """
        Resets the trajectory information and thus the character to 0,0,0 pointing to 0,0,1.

        Keyword Arguments:
            start_location {list} -- [description] (default: {[0,0,0]})
            start_orientation {list} -- [description] (default: {[1,0,0,0]})
            start_direction {list} -- [description] (default: {[0,0,1]})
        """
        self.traj_root_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_root)
        self.traj_root_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_root)
        self.traj_root_rotations = np.zeros(self.n_frames_tr_win_root)
        # TODO control rotations using these dirs
        # self.traj_root_directions = np.array([start_direction] * self.n_frames_tr_win_root)
        self.traj_root_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames_tr_win_root)

        ### Trajectory info of the hand
        # Wrist positions contain all 3 components
        self.traj_right_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)
        self.traj_right_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)

        # Wrist positions contain all 3 components
        self.traj_left_wrist_pos = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)
        self.traj_left_wrist_vels = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win_wrist)

        self.traj_reached_left_wrist = {'f': 0, 'r': 0}
        self.traj_reached_right_wrist = {'f': 0, 'r': 0}

        self.punch_completed_left = False
        self.punch_completed_right = False

        self.punch_half_completed_left = False
        self.punch_half_completed_right = False

        self.punch_frames_left = 0
        self.punch_frames_right = 0
        self.punch_frames_left_half_comp = 0
        self.punch_frames_right_half_comp = 0

        for idx in range(self.median_idx_root + 1):
            self.traj_root_pos[idx] = np.array(start_location)
            self.traj_root_rotations[idx] = np.array(start_orientation)
            self.traj_root_directions[idx] = np.array(start_direction)
