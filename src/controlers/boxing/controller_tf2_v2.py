"""author: Chirag Bhuvaneshwara """
import numpy as np
from src.nn.mann_keras_v2.mann import MANN
from ..abstract_controller import Controller
from .trajectory import Trajectory
from .character import Character
from ... import utils


class BoxingController(Controller):
    """
    This is a controller for punch target input.

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, network: MANN, data_config, dataset_npz_path, norm):  # xdim, ydim, end_joints = 5, numJoints = 21):

        # jd = config_store['joint_indices']
        # self.hand_left = jd['LeftWrist']
        # self.hand_right = jd['RightWrist']

        self.network = network
        self.xdim = network.input_size
        self.ydim = network.output_size

        self.endJoints = data_config["end_joints"]
        self.n_joints = data_config["num_joints"] + self.endJoints
        self.use_rotations = data_config["use_rotations"]
        self.n_gaits = data_config["n_gaits"]
        # TODO use_foot_contacts not needed as with new organization column indices can be handled automatically
        #  no matter what the x_input_frame or y_output_frame contains thanks to col_demarcation_ids
        self.use_foot_contacts = data_config["use_foot_contacts"]
        self.traj_window = data_config["window"]
        self.num_traj_samples = data_config["num_traj_samples"]
        self.traj_step = data_config["traj_step"]
        self.zero_posture = data_config["zero_posture"]
        self.bone_map = data_config["bone_map"]
        self.in_col_demarcation_ids = data_config["col_demarcation_ids"][0]
        self.out_col_demarcation_ids = data_config["col_demarcation_ids"][1]
        self.in_col_names = data_config["col_names"][0]
        self.out_col_names = data_config["col_names"][1]

        input_data = np.array([0.0] * self.xdim)
        out_data = np.array([0.0] * self.ydim)
        self.input = MANNInput(input_data, self.n_joints, self.endJoints, self.in_col_demarcation_ids)
        self.output = MANNOutput(out_data, self.n_joints, self.endJoints, self.bone_map, self.out_col_demarcation_ids)

        self.n_dims = 3
        self.num_targets = 2  # one for each hand

        self.punch_targets = np.array([0.0] * self.n_dims * self.num_targets)

        self.target_vel = np.array((0.0, 0.0, 0.0))
        self.target_dir = np.array((0.0, 0.0, 0.0))

        self.traj = Trajectory(data_config)
        self.char = Character(data_config)
        self.config_store = data_config
        self.dataset_npz_path = dataset_npz_path
        self.norm = norm
        self.__initialize()

    # TODO: Exp with punch target as curr wrist pos when not punching instead of 0 vector used in present implementation
    # TODO: Exp with only punch target and punch action and without any trajectory of the hands
    def pre_render(self, punch_targets, punch_labels, dir, space='local'):
        """
        This function is called before rendering. It prepares character and trajectory to be rendered.

        Arguments:
            @:param punch_targets {np.array(6)} --{left_punch_target,right_punch_target} user input punch target in local space
            @:param space: 'local' or 'global'
        Returns:

        """
        print('##############')
        print(dir)
        print('##############')

        #TODO: Supply direction in right hand coordinate system used in Python to pre_render instead of *-1
        direction = np.array(dir)*-1
        # direction = np.array([1, 1])
        # direction = np.array([0, 1])
        direction = utils.xz_to_x0yz(direction)
        # target_vel_speed = .0025 * np.linalg.norm(direction)
        target_vel_speed = 0.05 * np.linalg.norm(direction)
        self.target_vel = utils.glm_mix(self.target_vel, target_vel_speed * direction, 0.9)
        target_vel_dir = self.target_dir if utils.euclidian_length(self.target_vel) \
                                            < 1e-05 else utils.normalize(self.target_vel)
        self.target_dir = utils.mix_directions(self.target_dir, target_vel_dir, 0.9)

        # 2. Set new punch_phase and new punch target based on input
        prev_root_pos, prev_root_rot = self.traj.get_previous_pos_rot()

        right_p_target = np.array(punch_targets[:self.n_dims], dtype=np.float64)
        left_p_target = np.array(punch_targets[self.n_dims:], dtype=np.float64)
        if space == 'local':
            self.input.set_punch_target(right_p_target, left_p_target)
        elif space == 'global':
            if sum(right_p_target) != 0:
                right_p_target = right_p_target.reshape(1, len(right_p_target))
                right_p_target_local = self.char.convert_global_to_local(right_p_target, prev_root_pos, prev_root_rot,
                                                                         type_in='pos_hand')
                right_p_target = right_p_target_local.ravel()
            elif sum(left_p_target) != 0:
                left_p_target = left_p_target.reshape(1, len(left_p_target))
                left_p_target_local = self.char.convert_global_to_local(left_p_target, prev_root_pos, prev_root_rot,
                                                                        type_in='pos_hand')
                left_p_target = left_p_target_local.ravel()

            # print('#######################################################')
            # print('rtarget', right_p_target)
            # print('ltarget', left_p_target)
            # print('#######################################################')

            self.input.set_punch_target(right_p_target, left_p_target)
        else:
            raise ValueError("space variable accepts only local or global")

        # 3. Update/calculate trajectory based on input
        curr_right_p_label = np.array(punch_labels[0])
        curr_left_p_label = np.array(punch_labels[1])
        self.input.set_curr_punch_labels(curr_right_p_label, curr_left_p_label)

        right_shoulder_lp, left_shoulder_lp = self.output.get_shoulder_local_pos()
        #TODO: (Dont remember why) Update traj_labels only in post_render in update_from_predict i.e these 2 vecs are autoregressive ==> Similar to NSM
        self.traj.compute_future_wrist_trajectory(right_p_target, left_p_target, curr_right_p_label, curr_left_p_label,
                                                  right_shoulder_lp, left_shoulder_lp)

        self.traj.compute_future_root_trajectory(self.target_dir, self.target_vel)

        # 3. Set Trajectory input
        root_pos_tr, root_vels_tr, right_wrist_pos_tr, left_wrist_pos_tr, right_wrist_vels_tr, \
        left_wrist_vels_tr, right_labels_tr, left_labels_tr = self.traj.get_input(
            self.char.root_position, self.char.root_rotation)
        self.input.set_root_pos_tr(root_pos_tr)
        self.input.set_root_vels_tr(root_vels_tr)
        self.input.set_wrist_pos_tr(right_wrist_pos_tr, left_wrist_pos_tr)
        self.input.set_wrist_vels_tr(right_wrist_vels_tr, left_wrist_vels_tr)
        self.input.set_punch_labels_tr(right_labels_tr, left_labels_tr)

        # 4. Prepare and Set Joint Input
        # Steps 3 and 4 will update MANNInput Class completely to the NN's required input for current frame
        joint_pos, joint_vel = self.char.get_local_joint_pos_vel(prev_root_pos, prev_root_rot)
        self.input.set_local_pos(joint_pos.ravel())
        self.input.set_local_vel(joint_vel.ravel())

        # 5. Predict: Get MANN Output
        input_data = self.input.data.reshape(1, len(self.input.data))
        output_data = self.network.forward_pass(self.network, input_data, self.norm,
                                                [self.in_col_demarcation_ids, self.out_col_demarcation_ids])
        if np.isnan(output_data).any():
            raise Exception('Nans found in: ', np.argwhere(np.isnan(output_data)), '\n Input: ', input_data)

        self.output.data = output_data

        # 6. Process Prediction i.e. apply rotations and set traj foot drifting
        self.char.root_position, self.char.root_rotation = self.traj.get_world_pos_rot()

        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()

        # 7. Set character's new pose
        # TODO Try adding joint rotations in the input and output
        joint_rotations = []

        self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

        return self.input.data.ravel(), self.output.data.ravel()

    def post_render(self):
        """
        This function has to be called after rendering to prepare the next frame.

        Returns:
            float -- changed punch_phase depending on user output.
        """
        self.traj.step_forward(self.output.get_root_vel(), self.output.get_root_new_forward(),
                               self.output.get_wrist_local_vel(), self.input.get_curr_punch_labels())

        # 1. update and smooth trajectory
        #TODO: Update traj_labels only here i.e these 2 vecs are autoregressive ==> Similar to NSM
        self.traj.update_from_predict(self.output.get_next_traj(), self.input.get_curr_punch_labels())

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    def reset(self, start_location=np.array([0.0, 0.0, 0.0]), start_orientation=0,
              start_direction=np.array([0.0, 0.0, 0.0])):
        """
        Resets the controller to start location, orientation and direction.

        Arguments:
            start_location {[type_in]} -- [description]
            start_orientation {[type_in]} -- [description]
            start_direction {[type_in]} -- [description]
        """
        self.char.reset(start_location, start_orientation)
        # self.traj.reset(start_location, start_orientation, start_direction)
        self.traj.reset()
        self.__initialize()
        print('###################################')
        print('RESET DONE')
        print('###################################')

    def copy(self):
        """
        Should copy the controler. At the moment, just creates a new, blank controler.

        Returns:
            [type_in] -- [description]
        """
        return Controller(self.network, self.config_store)

    def get_pose(self):
        """
        This function forwards the posture for the next frame. Possibly just a forward from the character class

        Returns:
            np.array(float, (n_joints, 3)) - joint positions

        """

        # root_pos, root_rot = self.traj.get_world_pos_rot()
        root_pos, root_rot = np.array(self.char.root_position), np.array(self.char.root_rotation)
        # pos, vel = self.char.get_local_joint_pos_vel(root_pos, root_rot)
        pos = self.char.joint_positions
        return np.reshape(pos, (self.char.joints, 3))

    def get_trajectroy_for_vis(self):
        tr = self.traj
        step = self.traj_step
        # right_wr_tr, left_wr_tr = self.traj.get_global_arm_tr()
        right_wr_tr, left_wr_tr = tr.traj_right_wrist_pos[::step], tr.traj_left_wrist_pos[::step]
        # print(right_wr_tr)
        # print(left_wr_tr)
        root_tr = tr.traj_root_pos[::step]
        # print('-------------------')
        # print(root_tr)
        # print('-------------------')
        root_vels_tr = tr.traj_root_vels[::step]
        right_wr_vels_tr = tr.traj_right_wrist_vels[::step]
        left_wrist_vels_tr = tr.traj_left_wrist_vels[::step]
        return root_tr, root_vels_tr, right_wr_tr, left_wr_tr, right_wr_vels_tr, left_wrist_vels_tr

    def get_world_pos_rot(self):
        return np.array(self.char.root_position), float(self.char.root_rotation)

    def __initialize(self):
        self.traj.traj_left_punch_labels = np.zeros(self.traj.traj_left_punch_labels.shape)
        self.traj.traj_right_punch_labels = np.zeros(self.traj.traj_right_punch_labels.shape)

        # TODO Maybe Get only init local positions from a frame in the mocap data that you think is in neutral position.
        #  Get the rest of the variables from norm?
        data = np.load(self.dataset_npz_path)
        x_train = data["x"]
        y_train = data["y"]

        tmp_input = MANNInput(x_train[100].ravel(), self.n_joints, self.endJoints, self.in_col_demarcation_ids)
        tmp_output = MANNOutput(y_train[100].ravel(), self.n_joints, self.endJoints, self.bone_map,
                                self.out_col_demarcation_ids)

        self.input.data = np.array(self.norm["x_mean"], dtype=np.float64)
        self.output.data = np.array(self.norm["y_mean"], dtype=np.float64)

        # r_in, l_in = tmp_input.get_wrist_pos_traj()
        # self.input.set_wrist_pos_tr(r_in, l_in)
        # r_in, l_in = tmp_output.get_wrist_pos_traj()
        # self.output.set_wrist_pos_tr(r_in, l_in)

        # TODO Verify functioning and provide better variable names
        right, left = self.input.get_wrist_pos_traj()
        right, left = right.reshape(10, 3), left.reshape(10, 3)
        right, left = np.repeat(right, repeats=3, axis=0), np.repeat(left, repeats=3, axis=0)
        right = self.traj.convert_local_to_global(right, 'pos', arm='right')
        left = self.traj.convert_local_to_global(left, 'pos', arm='left')
        # TODO Set wrist positions to mean positions ==> Maybe take the positions from the mean position you have in
        #  norm
        self.traj.traj_right_wrist_pos, self.traj.traj_left_wrist_pos = right, left

        if self.use_rotations:
            # TODO Add joint rotations in input and output vectors
            raise ValueError("Joint rotations in input and output are not yet supported")
            # joint_rotations = self.output.getRotations()  # twist rotations
        else:
            joint_rotations = []

        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()

        self.char.set_pose(joint_positions, joint_velocities, joint_rotations, init=True)
        self.post_render()


###########################################################################################


class MANNInput(object):
    """
    This class is managing the network input. It is depending on the network data model

    Arguments:
        object {[type_in]} -- [description]

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, data, n_joints, end_joints, x_column_demarcation_ids):
        self.data = data
        self.joints = n_joints + end_joints
        # 3 dimensional space
        self.n_dims = 3
        self.col_demarcation_ids = x_column_demarcation_ids

    def __set_data__(self, key_name, data_sub_part):
        ids = self.col_demarcation_ids[key_name]
        key_data_start = ids[0]
        key_data_end = ids[1]
        self.data[key_data_start: key_data_end] = data_sub_part

    def __get_data__(self, key_name):
        ids = self.col_demarcation_ids[key_name]
        key_data_start = ids[0]
        key_data_end = ids[1]
        return self.data[key_data_start: key_data_end]

    def set_root_pos_tr(self, root_pos_traj):
        self.__set_data__("x_root_pos_tr", root_pos_traj)

    def set_root_vels_tr(self, root_vels_traj):
        self.__set_data__("x_root_vels_tr", root_vels_traj)

    def set_wrist_pos_tr(self, right_wrist_pos_traj, left_wrist_pos_traj):
        self.__set_data__("x_right_wrist_pos_tr", right_wrist_pos_traj)
        self.__set_data__("x_left_wrist_pos_tr", left_wrist_pos_traj)

    def set_wrist_vels_tr(self, right_wrist_vels_traj, left_wrist_vels_traj):
        self.__set_data__("x_right_wrist_vels_tr", right_wrist_vels_traj)
        self.__set_data__("x_left_wrist_vels_tr", left_wrist_vels_traj)

    def set_punch_labels_tr(self, right_punch_labels_traj, left_punch_labels_traj):
        self.__set_data__("x_right_punch_labels_tr", right_punch_labels_traj)
        self.__set_data__("x_left_punch_labels_tr", left_punch_labels_traj)

    def set_curr_punch_labels(self, right_label, left_label):
        self.__set_data__("x_right_punch_labels", right_label)
        self.__set_data__("x_left_punch_labels", left_label)

    def set_punch_target(self, right_target, left_target):
        self.__set_data__("x_right_punch_target", right_target)
        self.__set_data__("x_left_punch_target", left_target)

    def set_local_pos(self, pos):
        self.__set_data__("x_local_pos", pos)

    def set_local_vel(self, vel):
        self.__set_data__("x_local_vel", vel)

    def get_wrist_pos_traj(self):
        right_pos_traj = self.__get_data__('x_right_wrist_pos_tr')
        left_pos_traj = self.__get_data__('x_left_wrist_pos_tr')
        return right_pos_traj, left_pos_traj

    def get_curr_punch_labels(self):
        ph_r = self.__get_data__('x_right_punch_labels')
        ph_l = self.__get_data__('x_left_punch_labels')
        # TODO Why use a dict. Make it simpler
        return {'right': ph_r, 'left': ph_l}


class MANNOutput(object):
    """
    This class is managing the network output. It is depending on the network data model.

    Arguments:
        object {[type_in]} -- [description]

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, data, n_joints, end_joints, bone_map, y_column_demarcation_ids):
        self.data = data
        self.joints = n_joints + end_joints
        # 3 dimensional space
        self.num_coordinate_dims = 3
        self.endJoints = end_joints
        self.bone_map = bone_map
        self.col_demarcation_ids = y_column_demarcation_ids

    def __set_data__(self, key_name, data_sub_part):
        ids = self.col_demarcation_ids[key_name]
        key_data_start = ids[0]
        key_data_end = ids[1]
        self.data[key_data_start: key_data_end] = data_sub_part

    def __get_data__(self, key_name):
        ids = self.col_demarcation_ids[key_name]
        key_data_start = ids[0]
        key_data_end = ids[1]
        return self.data[key_data_start: key_data_end]

    def get_root_vel(self):
        return self.__get_data__('y_root_velocity')

    def get_root_new_forward(self):
        return self.__get_data__('y_root_new_forward')

    def get_rotational_vel(self):
        root_vel = self.get_root_vel()
        root_new_fwd_dir = self.get_root_new_forward()
        return np.concatenate((root_vel, root_new_fwd_dir))

    def get_root_rotation_velocity(self):
        root_new_fwd_x_z = self.get_root_new_forward()
        root_new_fwd_3d = utils.xz_to_x0yz(root_new_fwd_x_z)
        return utils.z_angle(root_new_fwd_3d)

    def get_curr_punch_labels(self):
        ph_r = self.__get_data__('y_right_punch_labels')
        ph_l = self.__get_data__('y_left_punch_labels')
        # TODO Why use a dict. Make it simpler
        return {'right': ph_r, 'left': ph_l}

    def get_foot_contacts(self):
        right_foot = self.__get_data__('y_foot_contacts')
        left_foot = self.__get_data__('y_foot_contacts')
        return np.concatenate((right_foot, left_foot))

    def get_root_pos_traj(self):
        return self.__get_data__('y_root_pos_tr')

    def get_root_vel_traj(self):
        return self.__get_data__('y_root_vels_tr')

    def get_wrist_pos_traj(self):
        right_pos_traj = self.__get_data__('y_right_wrist_pos_tr')
        left_pos_traj = self.__get_data__('y_left_wrist_pos_tr')
        return right_pos_traj, left_pos_traj

    def get_wrist_vels_traj(self):
        right_vels_traj = self.__get_data__('y_right_wrist_vels_tr')
        left_vels_traj = self.__get_data__('y_left_wrist_vels_tr')
        return right_vels_traj, left_vels_traj

    def get_punch_labels_traj(self):
        right_labels = self.__get_data__('y_right_punch_labels_tr')
        left_labels = self.__get_data__('y_left_punch_labels_tr')
        return right_labels, left_labels

    def get_local_pos(self):
        return self.__get_data__('y_local_pos')

    def get_local_vel(self):
        return self.__get_data__('y_local_vel')

    def get_wrist_local_vel(self):
        lv = self.__get_data__('y_local_vel')

        r_wr_start = self.bone_map["RightWrist"] * 3
        r_wr_lv = lv[r_wr_start: r_wr_start + 3]

        l_wr_start = self.bone_map["LeftWrist"] * 3
        l_wr_lv = lv[l_wr_start: l_wr_start + 3]

        return r_wr_lv, l_wr_lv

    def get_shoulder_local_pos(self):
        lp = self.__get_data__('y_local_pos')

        r_start = self.bone_map["RightShoulder"] * 3
        r_lp = lp[r_start: r_start + 3]

        l_start = self.bone_map["LeftShoulder"] * 3
        l_lp = lp[l_start: l_start + 3]

        return r_lp, l_lp

    def get_next_traj(self):
        rp_tr = self.get_root_pos_traj()
        rv_tr = self.get_root_vel_traj()
        rwp_tr, lwp_tr = self.get_wrist_pos_traj()
        rwv_tr, lwv_tr = self.get_wrist_vels_traj()
        rpunch_tr, lpunch_tr = self.get_punch_labels_traj()
        # pred_dir = self.get_root_new_forward()

        return rp_tr, rv_tr, rwp_tr, lwp_tr, rwv_tr, lwv_tr, rpunch_tr, lpunch_tr
        # , pred_dir

    def set_wrist_pos_tr(self, right_wrist_pos_traj, left_wrist_pos_traj):
        self.__set_data__("y_right_wrist_pos_tr", right_wrist_pos_traj)
        self.__set_data__("y_left_wrist_pos_tr", left_wrist_pos_traj)
