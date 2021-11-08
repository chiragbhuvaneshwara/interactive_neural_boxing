"""author: Chirag Bhuvaneshwara """
import numpy as np
import pandas as pd

from train.nn.mann_keras.mann import MANN
from vis.backend.controller.boxing.trajectory import Trajectory
from vis.backend.controller.boxing.character import Character
from vis.backend.controller.boxing.mann_input import MANNInput
from vis.backend.controller.boxing.mann_output import MANNOutput
from vis.backend.controller import utils


class BoxingController:
    """
    This is a controller for controlling boxing motions such as punching by inputting info such as punch target.
    """

    def __init__(self, network: MANN, data_config, norm):

        self.network = network
        self.xdim = network.input_size
        self.ydim = network.output_size

        self.data_config = data_config
        self.n_joints = data_config["num_joints"]
        self.traj_window_wrist = data_config["traj_window_wrist"]
        self.traj_window_root = data_config["traj_window_root"]
        self.num_traj_samples_wrist = data_config["num_traj_samples_wrist"]
        self.num_traj_samples_root = data_config["num_traj_samples_root"]
        # self.traj_step_root = data_configuration['traj_step_root']  # 5
        # self.traj_step_wrist = data_configuration['traj_step_wrist']  # 5
        self.traj_step_root = data_config['frame_rate_div']  # 5
        self.traj_step_wrist = data_config['frame_rate_div']  # 5
        self.zero_posture = data_config["zero_posture"]
        self.bone_map = data_config["bone_map"]
        self.in_col_demarcation_ids = data_config["col_demarcation_ids"][0]
        self.out_col_demarcation_ids = data_config["col_demarcation_ids"][1]
        self.in_col_names = data_config["col_names"][0]
        self.out_col_names = data_config["col_names"][1]
        self.num_gating_experts = data_config["num_gating_experts"]

        input_data = np.array([0.0] * self.xdim)
        out_data = np.array([0.0] * self.ydim)
        self.input = MANNInput(input_data, self.n_joints, self.in_col_demarcation_ids)
        self.output = MANNOutput(out_data, self.n_joints, self.bone_map, self.out_col_demarcation_ids)

        self.n_dims = 3
        self.num_targets = 2  # one for each hand

        self.target_vel = np.array((0.0, 0.0, 0.0))
        self.target_dir = np.array((0.0, 0.0, 1.0))

        self.traj = Trajectory(data_config)
        self.char = Character(data_config)
        self.dataset_npz_path = data_config["dataset_npz_path"]
        self.norm = norm
        # self.user_dir = np.array([0, 0])
        # self.facing_dir = np.array([1, 0, 0])
        self.__initialize()
        self.punch_targets = [0] * 6

        column_names = [k for k in self.bone_map.keys()]
        column_names = [c + "_" + v + "_" + str(i) for v in ["positions", "vels"] for c in column_names for i in
                        range(3)]
        column_names = column_names + [c_name + hand + "_" + str(i) for c_name in ["punch_target_"] for hand in
                                       ["right", "left"] for i in
                                       range(3)]
        self.eval_column_names = column_names + [c_name + hand for c_name in
                                                 ["punch_half_complete_", "punch_complete_", "punch_frames_"] for hand
                                                 in
                                                 ["right", "left"]]
        self.eval_df = pd.DataFrame(columns=self.eval_column_names)

    def pre_render(self, punch_targets, punch_labels, dir, space='local'):
        """
        This method is called before rendering. It prepares character and trajectory to be rendered.

        @param: punch_targets {np.array(6)} --{left_punch_target,right_punch_target} user input punch target
        @param: punch_labels {list(2)} --{left_punch_label,right_punch_label} user input punch label generated based
        on input punch_target
        @param: dir {list(2)} --{x and z co-ordinates} direction of motion intended by the user
        @param: traj_reached int number of trajectory future trajectory pts that have reached the target
        @param: space: 'local' or 'global', indicating the co-ordinate system of the punch target data
        @return:
        input and output data numpy arrays generated by the controller for the next frame
        """

        #################### Pre-Predict#################################
        self.punch_targets = punch_targets
        direction = np.array(dir)
        # self.user_dir = direction[:]
        # direction = np.array([1,0])
        direction = utils.xz_to_x0yz(direction)
        # target_vel_speed = 0.035 * np.linalg.norm(direction)
        # TODO Finalize one of the below values for target_vel_speed
        # target_vel_speed = 0.02 * np.linalg.norm(direction)
        target_vel_speed = 0.015 * np.linalg.norm(direction)
        # target_vel_speed = 0.005 * np.linalg.norm(direction)
        self.target_vel = utils.glm_mix(self.target_vel, target_vel_speed * direction, 0.4)
        target_vel_dir = self.target_dir if utils.euclidian_length(self.target_vel) \
                                            < 1e-05 else utils.normalize(self.target_vel)
        self.target_dir = utils.mix_directions(self.target_dir, target_vel_dir, 0.9)

        # 2. Set new punch_label and new punch target based on input
        prev_root_pos, prev_root_rot = self.traj.get_previous_pos_rot()

        right_p_target = np.array(punch_targets[:self.n_dims], dtype=np.float64)
        left_p_target = np.array(punch_targets[self.n_dims:], dtype=np.float64)
        if space == 'local':
            self.input.set_punch_target(right_p_target, left_p_target)
        elif space == 'global':
            right_p_target_local = np.array([0, 0, 0])
            left_p_target_local = np.array([0, 0, 0])
            if sum(right_p_target) != 0:
                right_p_target = right_p_target.reshape(1, len(right_p_target))
                right_p_target_local = self.char.convert_global_to_local(right_p_target, prev_root_pos, prev_root_rot,
                                                                         type_in='pos_hand')
                right_p_target_local = right_p_target_local.ravel()
                right_p_target = right_p_target.ravel()
            elif sum(left_p_target) != 0:
                left_p_target = left_p_target.reshape(1, len(left_p_target))
                left_p_target_local = self.char.convert_global_to_local(left_p_target, prev_root_pos, prev_root_rot,
                                                                        type_in='pos_hand')
                left_p_target_local = left_p_target_local.ravel()
                left_p_target = left_p_target.ravel()

            self.input.set_punch_target(right_p_target_local, left_p_target_local)
        else:
            raise ValueError("space variable accepts only local or global")

        curr_right_p_label = np.array(punch_labels[0])
        curr_left_p_label = np.array(punch_labels[1])

        self.input.set_curr_punch_labels(curr_right_p_label, curr_left_p_label)

        # 3. Update/calculate trajectory based on input
        right_shoulder_lp, left_shoulder_lp = self.output.get_shoulder_local_pos()
        right_wrist_lp, left_wrist_lp = self.output.get_wrist_local_pos()
        self.traj.compute_future_wrist_trajectory(right_p_target, left_p_target, right_shoulder_lp, left_shoulder_lp,
                                                  right_wrist_lp, left_wrist_lp, )

        self.traj.compute_future_root_trajectory(self.target_dir, self.target_vel)

        # 3. Set Trajectory input
        root_pos_tr, root_vels_tr, root_dirs_tr, right_wrist_pos_tr, left_wrist_pos_tr, right_wrist_vels_tr, \
        left_wrist_vels_tr = self.traj.get_input(
            self.char.root_position, self.char.root_rotation)
        self.input.set_root_pos_tr(root_pos_tr)
        self.input.set_root_vels_tr(root_vels_tr)
        self.input.set_root_dirs_tr(root_dirs_tr)
        self.input.set_wrist_pos_tr(right_wrist_pos_tr, left_wrist_pos_tr)
        # self.input.set_wrist_vels_tr(right_wrist_vels_tr, left_wrist_vels_tr)
        # self.input.set_punch_labels_tr(right_labels_tr, left_labels_tr)

        # 4. Prepare and Set Joint Input
        # Steps 3 and 4 will update MANNInput Class completely to the NN's required input for current frame
        joint_pos, joint_vel = self.char.get_local_joint_pos_vel(prev_root_pos, prev_root_rot)
        self.input.set_local_pos(joint_pos.ravel())
        self.input.set_local_vel(joint_vel.ravel())

        # 5. Predict: Get MANN Output
        input_data = self.input.data.reshape(1, len(self.input.data))
        #################### Pre-Predict#################################

        #################### Predict #################################
        output_data = self.network.forward_pass(self.network, input_data, self.norm, self.num_gating_experts)
        if np.isnan(output_data).any():
            raise Exception('Nans found in: ', np.argwhere(np.isnan(output_data)), '\n Input: ', input_data)

        self.output.data = output_data
        #################### Predict ################################

        #################### Pre-Render#################################

        # 6. Process Prediction i.e. apply rotations and set traj foot drifting
        self.char.root_position, self.char.root_rotation = self.traj.get_world_pos_rot()

        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()
        foot_contacts = self.output.get_foot_contacts()

        foot_contacts[foot_contacts < 0.3] = 0
        foot_drifting = self.char.compute_foot_sliding(joint_positions, joint_velocities, foot_contacts)
        self.traj.foot_drifting = foot_drifting

        # 7. Set character's new pose
        joint_rotations = []

        self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

        #################### Pre-Render#################################

        return self.input.data.ravel(), self.output.data.ravel()

    def post_render(self):
        """
        This method has to be called after rendering to prepare the next frame.
        """

        self.traj.step_forward(self.output.get_root_vel(), self.output.get_root_new_forward(),
                               self.output.get_wrist_local_vel(), self.output.get_wrist_local_pos(),
                               self.output.get_root_local_pos(), self.input.get_curr_punch_labels())

        # 1. update and smooth trajectory
        self.traj.update_from_predict(self.output.get_next_traj())

    def reset(self, start_location=np.array([0.0, 0.0, 0.0]), start_orientation=0,
              start_direction=np.array([0.0, 0.0, 1]), empty_df=False):
        """
        Resets the controller to start location, orientation and direction.

        @param start_location: np arr(3)
        @param start_orientation: float, rotation info in radian
        @param start_direction: np arr(3)
        """
        self.char.reset(start_location, start_orientation)
        self.traj.reset(start_location, start_orientation, start_direction)
        self.__initialize()
        if empty_df:
            self.eval_df = pd.DataFrame(columns=self.eval_column_names)
        print('###################################')
        print('RESET DONE')
        print('###################################')

    def get_pose(self):
        """
        This method forwards the posture for the next frame. Mostly just a forward of the joint positions
         from the character class.

        @return:
        np.array(float, (n_joints, 3)) - joint positions in global co-ordinate space
        """

        pos = self.char.joint_positions
        return np.reshape(pos, (self.char.joints, 3))

    def get_trajectroy_for_vis(self):
        """
        This method fetches from the Trajectory class and returns the trajectory variables that will be visualized on
        Unity.

        @return:
         root_tr: np.arr(num_root_tr_pts, 3), trajectory of root positions
         root_vels_tr: np.arr(num_root_tr_pts, 3), trajectory of root velocities
         right_wrist_tr: np.arr(num_right_wrist_tr_pts, 3), trajectory of right hand's wrist positions
         left_wrist_tr: np.arr(num_left_wrist_tr_pts, 3), trajectory of left hand's wrist positions
         right_wrist_vels_tr: np.arr(num_right_wrist_tr_pts, 3), trajectory of right hand's wrist velocities
         right_left_vels_tr: np.arr(num_left_wrist_tr_pts, 3), trajectory of left hand's wrist velocities
        """
        tr = self.traj
        step_wrist = self.traj_step_wrist
        step_root = self.traj_step_root
        right_wr_tr, left_wr_tr = tr.traj_right_wrist_pos[::step_wrist], tr.traj_left_wrist_pos[::step_wrist]
        root_tr = tr.traj_root_pos[::step_root]
        root_vels_tr = tr.traj_root_vels[::step_root]
        right_wr_vels_tr = tr.traj_right_wrist_vels[::step_wrist]
        left_wrist_vels_tr = tr.traj_left_wrist_vels[::step_wrist]
        return root_tr, root_vels_tr, right_wr_tr, left_wr_tr, right_wr_vels_tr, left_wrist_vels_tr

    def get_world_pos_rot(self):
        """
        Simple method that accesses the character class to return the character's position in global co-ordinate space
        and the character's rotation w.r.t the forward direction.

        @return:
        np.array(n_joints, 3) - joint positions in global co-ordinate space
        np.array(n_joints) - joint rotations w.r.t the forward direction
        """
        return np.array(self.char.root_position), float(self.char.root_rotation)

    def get_punch_metrics(self, hand):
        if hand == "right":
            punch_target = np.array(self.punch_targets[:self.n_dims], dtype=np.float64)
        else:
            punch_target = np.array(self.punch_targets[self.n_dims:], dtype=np.float64)

        pm = self.char.compute_punch_metrics(hand, punch_target)
        return pm

    def eval_values(self, record=False, save=False, save_path=None):
        if record:
            c = self.char
            t = self.traj
            new_row = [
                list(c.joint_positions.ravel()) + list(c.joint_velocities.ravel())
                + self.punch_targets
                + [t.punch_half_completed_right, t.punch_half_completed_left]
                + [t.punch_completed_right, t.punch_completed_left]
                + [t.punch_frames_right, t.punch_frames_left]
            ]

            if t.punch_completed_left or t.punch_completed_right:
                self.reset()

            eval_df_new_row = pd.DataFrame(new_row, columns=self.eval_column_names)
            self.eval_df = self.eval_df.append(eval_df_new_row, ignore_index=True)

        elif save:
            assert save_path is not None
            self.eval_df.to_csv(save_path)

    def __initialize(self, init_type="mean", init_tr_wrist=True):
        """
        This method needs to be called at the beginning i.e when the controller is initialized. It initializes the
        input and output data to the network with either the mean, from a specific row in the dataset or a mix of both.
        It also sets the initial pose of the character and performs a post_render operation.

        @param: init_type: str, one of "mean", "dataset" or "both"
        @param: init_tr_wrist: bool, if True, initializes the trajectory wrist positions from the instance of MANNInput
        that was first initialized in this method.
        """

        data = np.load(self.dataset_npz_path)
        x_train = data["x"]
        y_train = data["y"]
        dataset_input = MANNInput(x_train[100].ravel(), self.n_joints, self.in_col_demarcation_ids)
        dataset_output = MANNOutput(y_train[100].ravel(), self.n_joints, self.bone_map,
                                    self.out_col_demarcation_ids)
        if init_type == "dataset":
            self.input.data = dataset_input.data
            self.output.data = dataset_output.data

        elif init_type == "mean" or init_type == "both":
            self.input.data = np.array(self.norm["x_mean"], dtype=np.float64)
            self.output.data = np.array(self.norm["y_mean"], dtype=np.float64)

        if init_type == "both":
            r_in, l_in = dataset_input.get_wrist_pos_traj()
            self.input.set_wrist_pos_tr(r_in, l_in)
            r_in, l_in = dataset_output.get_wrist_pos_traj()
            self.output.set_wrist_pos_tr(r_in, l_in)

        if init_tr_wrist:
            right_pos, left_pos = self.input.get_wrist_pos_traj()
            right_pos, left_pos = right_pos.reshape(self.num_traj_samples_wrist, 3), left_pos.reshape(
                self.num_traj_samples_wrist,
                3)
            right_pos, left_pos = np.repeat(right_pos, repeats=self.traj_step_wrist, axis=0), np.repeat(left_pos,
                                                                                                        repeats=self.traj_step_wrist,
                                                                                                        axis=0)
            right_pos = self.traj.convert_local_to_global(right_pos, 'pos', arm='right')
            left_pos = self.traj.convert_local_to_global(left_pos, 'pos', arm='left')
            self.traj.traj_right_wrist_pos, self.traj.traj_left_wrist_pos = right_pos, left_pos

        joint_rotations = []
        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()

        self.char.set_pose(joint_positions, joint_velocities, joint_rotations, init=True)
        self.post_render()
