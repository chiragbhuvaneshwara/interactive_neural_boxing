"""author: Chirag Bhuvaneshwara """
import numpy as np
import math, time
import pandas as pd
from src.nn.keras_mods.mann_keras import MANN
from ..controller import Controller
from .trajectory_2 import Trajectory
from .character_box import Character
from ... import utils

DEBUG = False
DEBUG_TIMING = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)


class BoxingController(Controller):
    """
    This is a controller for punch target input.

    Returns:
        [type] -- [description]
    """

    def __init__(self, network: MANN, config_store):  # xdim, ydim, end_joints = 5, numJoints = 21):

        # jd = config_store['joint_indices']
        # self.hand_left = jd['LeftWrist']
        # self.hand_right = jd['RightWrist']

        self.network = network
        self.xdim = network.input_dim
        self.ydim = network.output_dim

        self.endJoints = config_store["endJoints"]
        self.n_joints = config_store["numJoints"] + self.endJoints
        self.use_rotations = config_store["use_rotations"]
        self.n_gaits = config_store["n_gaits"]
        self.use_foot_contacts = config_store["use_footcontacts"]
        self.traj_window = config_store["window"]
        self.num_traj_samples = config_store["num_traj_samples"]
        self.traj_step = config_store["traj_step"]
        self.zero_posture = config_store["zero_posture"]
        self.joint_names_ids = config_store["joint_indices"]
        self.in_col_names_ids = config_store["col_indices"][0]
        self.out_col_names_ids = config_store["col_indices"][1]
        self.in_col_names = config_store["col_names"][0]
        self.out_col_names = config_store["col_names"][1]

        input_data = np.array([0.0] * self.xdim)
        out_data = np.array([0.0] * self.ydim)
        self.input = MANNInput(input_data, self.n_joints, self.endJoints, self.joint_names_ids, self.in_col_names_ids)
        self.output = MANNOutput(out_data, self.n_joints, self.endJoints, self.use_foot_contacts, self.joint_names_ids,
                                 self.out_col_names_ids)

        self.previous_punch_phase = {'left': 0, 'right': 0}
        self.previous_joint_pos = np.array([[0.0, 0.0, 0.0]] * self.n_joints)
        self.previous_joint_vel = np.array([[0.0, 0.0, 0.0]] * self.n_joints)

        self.n_dims = 3
        self.num_targets = 2  # one for each hand

        self.punch_targets = np.array([0.0] * self.n_dims * self.num_targets)

        self.target_vel = np.array((0.0, 0.0, 0.0))
        self.target_dir = np.array((0.0, 0.0, 0.0))

        self.traj = Trajectory(config_store)
        self.char = Character(config_store)
        self.config_store = config_store
        self.__initialize()

    def pre_render(self, punch_targets, space='local'):
        """
        This function is called before rendering. It prepares character and trajectory to be rendered.

        Arguments:
            punch_targets {np.array(6)} --{left_punch_target,right_punch_target} user input punch target in local space
            space: 'local' or 'global'
        Returns:

        """
        # User input direction but for simplicity using predicted direction
        # direction = self.output.get_root_new_forward()
        direction = np.array([0, 0])
        direction = utils.convert_to_zero_y_3d(direction)
        # target_vel_speed = .00025 * np.linalg.norm(direction)
        target_vel_speed = np.linalg.norm(direction)
        self.target_vel = utils.glm_mix(self.target_vel, target_vel_speed * direction, 0.9)
        # self.target_vel = utils.convert_to_zero_y_3d(self.output.get_root_vel())
        target_vel_dir = self.target_dir if utils.euclidian_length(self.target_vel) \
                                            < 1e-05 else utils.normalize(self.target_vel)
        self.target_dir = utils.mix_directions(self.target_dir, target_vel_dir, 0.9)

        # 3. Update/calculate trajectory based on input
        right_pos_traj_target, left_pos_traj_target = self.output.get_wrist_pos_traj()
        right_vels_traj_target, left_vels_traj_target = self.output.get_wrist_vels_traj()
        self.traj.compute_future_wrist_trajectory(right_pos_traj_target, right_vels_traj_target, left_pos_traj_target,
                                                  left_vels_traj_target)

        pred_pos_traj = self.output.get_root_pos_traj()
        pred_vels_traj = self.output.get_root_vel_traj()
        pred_fwd_dir_target = self.output.get_root_new_forward()
        self.traj.compute_future_root_trajectory(self.target_dir, self.target_vel)

        # 3. Set Trajectory input
        input_root_pos, input_root_vels, input_right_wrist_pos, input_right_wrist_vels, \
        input_left_wrist_pos, input_left_wrist_vels = self.traj.get_input(
            self.char.root_position, self.char.root_rotation)
        self.input.set_rootpos_traj(input_root_pos)
        self.input.set_rootvel_traj(input_root_vels)
        self.input.set_wrist_pos_tr(input_right_wrist_pos, input_left_wrist_pos)
        self.input.set_wrist_vels_tr(input_right_wrist_vels, input_left_wrist_vels)

        # 4. Prepare and Set Joint Input
        # Steps 3 and 4 will update MANNInput Class completely to the NN's required input for current frame
        punch_phase_r, punch_phase_l = self.get_previous_punch_phase()
        self.input.set_punch_phase(punch_phase_r, punch_phase_l)

        prev_root_pos, prev_root_rot = self.traj.getPreviousPosRot()
        joint_pos, joint_vel = self.char.getLocalJointPosVel(prev_root_pos, prev_root_rot)
        self.input.set_local_pos(joint_pos.ravel())
        self.input.set_local_vel(joint_vel.ravel())

        # 2. Set new punch_phase and new punch target based on input
        right_p_target = np.array(punch_targets[:self.n_dims])
        left_p_target = np.array(punch_targets[self.n_dims:])
        if space == 'local':
            self.input.set_punch_target(right_p_target, left_p_target)
        elif space == 'global':
            right_p_target = right_p_target.reshape(1, len(right_p_target))
            left_p_target = left_p_target.reshape(1, len(left_p_target))

            right_p_target_local = self.char.convert_global_to_local(right_p_target, prev_root_pos, prev_root_rot,
                                                                     type='pos')
            left_p_target_local = self.char.convert_global_to_local(left_p_target, prev_root_pos, prev_root_rot,
                                                                    type='pos')

            right_p_target = right_p_target_local.ravel()
            left_p_target = left_p_target_local.ravel()
            if sum(punch_targets) != 0:
                # print('#######################################################')
                # print('Converted punch targets')
                # print(right_p_target_local)
                print('r', right_p_target_local)
                print('l', left_p_target_local)
                # print('#######################################################')

            self.input.set_punch_target(right_p_target, left_p_target)
        else:
            raise ValueError("space variable accepts only local or global")

        # 5. Predict: Get MANN Output
        # in_data_df = pd.read_csv(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\data\boxing_fr_1_25\X.csv')
        # input_data = in_data_df.iloc[[164]].values
        # input_data = np.delete(input_data, 0, 1)
        # self.input.data = input_data.ravel()
        input_data = self.input.data.reshape(1, len(self.input.data))
        output_data = self.network.forward_pass(self.network, input_data, self.network.norm)
        if np.isnan(output_data).any():
            raise Exception('Nans found in: ', np.argwhere(np.isnan(output_data)), '\n Input: ', input_data)

        self.output.data = output_data.numpy().ravel()
        # self.output.data[4] = self.fix_phase(self.output.data[4])
        # self.output.data[5] = self.fix_phase(self.output.data[5])
        # print(self.output.data[4:6])

        # 6. Process Prediction i.e. apply rotations and set traj foot drifting
        self.char.root_position, self.char.root_rotation = self.traj.getWorldPosRot()

        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()

        # 7. Set character's new pose
        joint_rotations = []
        # TODO: Check if to apply before or after set_pose
        # self.char.root_rotation = self.output.get_root_rotation_velocity()  # [-2*pi, +2*pi]
        # self.char.root_position += utils.convert_to_zero_y_3d(self.output.get_root_vel())  # inserting 0 to Y axis
        self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

        return self.input.data.ravel(), self.output.data.ravel()

    def post_render(self):
        """
        This function has to be called after rendering to prepare the next frame.

        Returns:
            float -- changed punch_phase depending on user output.
        """
        if DEBUG:
            print("\n\n############## POST RENDER ###########")
        if DEBUG_TIMING:
            start_time = time.time()

        self.traj.step_forward(self.output.get_root_vel(), self.output.get_root_new_forward(),
                               self.output.get_wrist_vels_traj())

        # 1. update and smooth trajectory
        self.traj.update_from_predict(self.output.get_next_traj())

        # 2. update variables that'll be used for next input frame
        # pred_d_ph = self.output.get_punch_change_in_phase()
        # for hand in ['right', 'left']:
        #     self.previous_punch_phase[hand] = (self.previous_punch_phase[hand] + pred_d_ph[hand]) % 1.0
        # print(self.previous_punch_phase)

        pred_ph = self.output.get_punch_phase()
        for hand in ['right', 'left']:
            # self.previous_punch_phase[hand] = self.fix_phase(pred_ph[hand])
            self.previous_punch_phase[hand] = pred_ph[hand]

        # print('-------------------------------------------------------------')
        # print(self.previous_punch_phase)
        # print('-------------------------------------------------------------')

        self.previous_joint_pos = self.output.get_local_pos()
        self.previous_joint_vel = self.output.get_local_vel()

        return self.previous_punch_phase, self.previous_joint_pos, self.previous_joint_vel

    def reset(self, start_location, start_orientation, start_direction):
        """
        Resets the controller to start location, orientation and direction.

        Arguments:
            start_location {[type]} -- [description]
            start_orientation {[type]} -- [description]
            start_direction {[type]} -- [description]
        """
        self.char.reset(start_location, start_orientation)
        self.traj.reset(start_location, start_orientation, start_direction)

    def copy(self):
        """
        Should copy the controler. At the moment, just creates a new, blank controler.

        Returns:
            [type] -- [description]
        """
        return Controller(self.network, self.config_store)

    def getPose(self):
        """
        This function forwards the posture for the next frame. Possibly just a forward from the character class

        Returns:
            np.array(float, (n_joints, 3)) - joint positions

        """

        # root_pos, root_rot = self.traj.getWorldPosRot()
        root_pos, root_rot = np.array(self.char.root_position, dtype=np.float64), np.array(self.char.root_rotation,
                                                                                           dtype=np.float64)
        pos, vel = self.char.getLocalJointPosVel(root_pos, root_rot)
        return np.reshape(pos, (self.char.joints, 3))

    def getArmTrajectroy(self):
        # right_wr_tr, left_wr_tr = self.traj.get_global_arm_tr()
        right_wr_tr, left_wr_tr = self.traj.traj_right_wrist_positions[::self.traj_step], self.traj.traj_left_wrist_positions[::self.traj_step]
        return right_wr_tr, left_wr_tr

    def getGlobalRoot(self):
        # grt = self.traj.get_global_root_tr()
        grt = self.traj.traj_positions[::self.traj_step]
        gr = np.array(self.char.root_position)
        print('root_tr')
        print(grt[0])
        # return grt[1], gr
        print('--------')
        print(grt.shape)
        print(self.traj.traj_left_wrist_positions[::self.traj.traj_step])
        return grt, grt

    def getWorldPosRot(self):
        return np.array(self.char.root_position), float(self.char.root_rotation)

    def __initialize(self):
        # self.output.data = np.array(self.network.norm["Ymean"])  # * self.network.norm["Ystd"] + self.network.norm["Ymean"]
        # self.input.data = np.array(self.network.norm["Xmean"])  # * self.network.norm["Xstd"] + self.network.norm["Xmean"]

        self.input.data = np.array(self.network.norm["Xmean"], dtype=np.float64)
        self.output.data = np.array(self.network.norm["Ymean"], dtype=np.float64)

        if self.use_rotations:
            raise ValueError("Use Rotations not yet supported")
            # joint_rotations = self.output.getRotations()  # twist rotations
        else:
            joint_rotations = []

        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()

        self.previous_joint_pos = self.output.get_local_pos()
        self.previous_joint_vel = self.output.get_local_vel()

        self.char.set_pose(joint_positions, joint_velocities, joint_rotations, init=True)
        self.post_render()

        # # 7. set new character pose
        # # TODO: Check if to apply before or after set_pose
        # self.char.root_rotation = self.output.get_root_rotation_velocity()  # [-2*pi, +2*pi]
        # # self.char.root_rotation = self.output.get_root_rotation_velocity()  # [-2*pi, +2*pi]
        #
        # self.char.root_position += utils.convert_to_zero_y_3d(self.output.get_root_vel())
        # self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

    def get_previous_local_joint_pos_vel(self):
        return self.previous_joint_pos, self.previous_joint_vel

    def get_previous_punch_phase(self):
        return self.previous_punch_phase['right'], self.previous_punch_phase['left']

    def fix_phase(self, phase):
        if phase <= -1.5:
            phase = -1
        elif phase >= 1.5:
            phase = 1
        else:
            phase = round(phase)
        return phase


###########################################################################################


class MANNInput(object):
    """
    This class is managing the network input. It is depending on the network data model

    Arguments:
        object {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, data, n_joints, end_joints, joint_indices, x_col_names_ids):
        self.data = data
        self.joints = n_joints + end_joints
        # 3 dimensional space
        self.n_dims = 3
        self.joint_indices_dict = joint_indices
        self.col_names_ids = x_col_names_ids

    def __set_data__(self, data_part, id_start, id_end):
        self.data[id_start: id_end] = data_part

    def __set_data__(self, key_name, data_sub_part):
        ids = self.col_names_ids[key_name]
        key_data_start = ids[0]
        key_data_end = ids[1]
        self.data[key_data_start: key_data_end] = data_sub_part

    def set_rootpos_traj(self, rootpos_traj):
        self.__set_data__("x_rootposs_tr", rootpos_traj)
        # rootpos_traj_id = self.col_names_ids["x_rootposs_tr"]
        # self.__set_data__(rootpos_traj, rootpos_traj_id[0], rootpos_traj_id[1])

    def set_rootvel_traj(self, rootvel_traj):
        self.__set_data__("x_rootvels_tr", rootvel_traj)
        # rootvel_traj_id = self.col_names_ids["x_rootvels_tr"]
        # self.__set_data__(rootvel_traj, rootvel_traj_id[0], rootvel_traj_id[1])

    def set_wrist_pos_tr(self, right_wrist_pos_traj, left_wrist_pos_traj):
        self.__set_data__("x_right_wrist_pos_tr", right_wrist_pos_traj)
        self.__set_data__("x_left_wrist_pos_tr", left_wrist_pos_traj)

        # right_wrist_pos_ids = self.col_names_ids["x_right_wrist_pos_tr"]
        # left_wrist_pos_ids = self.col_names_ids["x_left_wrist_pos_tr"]
        # self.__set_data__(wrist_pos_traj, right_wrist_pos_ids[0], left_wrist_pos_ids[1])

    def set_wrist_vels_tr(self, right_wrist_vels_traj, left_wrist_vels_traj):
        self.__set_data__("x_right_wrist_vels_tr", right_wrist_vels_traj)
        self.__set_data__("x_left_wrist_vels_tr", left_wrist_vels_traj)

        # right_wrist_vels_ids = self.col_names_ids["x_right_wrist_vels_tr"]
        # left_wrist_vels_ids = self.col_names_ids["x_left_wrist_vels_tr"]
        # self.__set_data__(wrist_vels_traj, right_wrist_vels_ids[0], left_wrist_vels_ids[1])

    def set_punch_phase(self, right_phase, left_phase):
        curr_phase = np.array([right_phase, left_phase], dtype=np.float64)
        self.__set_data__("x_punch_phase", curr_phase)

        # punch_phase_id = self.col_names_ids["x_punch_phase"]
        # self.__set_data__(curr_phase, punch_phase_id[0], punch_phase_id[1])

    def set_punch_target(self, right_target, left_target):
        self.__set_data__("x_right_punch_target", right_target)
        self.__set_data__("x_left_punch_target", left_target)

        # punch_right_target_id = self.col_names_ids["x_right_punch_target"]
        # punch_left_target_id = self.col_names_ids["x_left_punch_target"]
        # punch_both_target_id = [punch_right_target_id[0], punch_left_target_id[1]]
        # self.__set_data__(targets, punch_both_target_id[0], punch_both_target_id[1])

    def set_local_pos(self, pos):
        self.__set_data__("x_local_pos", pos)

        # local_pos_id = self.col_names_ids["x_local_pos"]
        # self.__set_data__(pos, local_pos_id[0], local_pos_id[1])

    def set_local_vel(self, vel):
        self.__set_data__("x_local_vel", vel)

        # local_vel_id = self.col_names_ids["x_local_vel"]
        # self.__set_data__(vel, local_vel_id[0], local_vel_id[1])


class MANNOutput(object):
    """
    This class is managing the network output. It is depending on the network data model.

    Arguments:
        object {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, data, n_joints, end_joints, use_foot_contacts, joint_names_ids,
                 y_col_names_ids):
        self.data = data
        self.joints = n_joints + end_joints
        # 3 dimensional space
        self.num_coordinate_dims = 3
        self.endJoints = end_joints
        self.use_foot_contacts = use_foot_contacts
        self.joint_names_ids = joint_names_ids
        self.col_names_ids = y_col_names_ids

    def __get_data__(self, key_name):
        ids = self.col_names_ids[key_name]
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
        root_new_fwd_3d = utils.convert_to_zero_y_3d(root_new_fwd_x_z)
        return utils.z_angle(root_new_fwd_3d)

    def get_punch_change_in_phase(self):
        d_ph_r, d_ph_l = self.__get_data__('y_punch_dphase')
        return {'right': d_ph_r, 'left': d_ph_l}

    def get_punch_phase(self):
        ph_r, ph_l = self.__get_data__('y_punch_phase')
        return {'right': ph_r, 'left': ph_l}

    def get_foot_contacts(self):
        return self.__get_data__('y_foot_contacts')

    def get_root_pos_traj(self):
        return self.__get_data__('y_rootposs_tr')

    def get_root_vel_traj(self):
        return self.__get_data__('y_rootvels_tr')

    def get_wrist_pos_traj(self):
        right_pos_traj = self.__get_data__('y_right_wrist_pos_tr')
        left_pos_traj = self.__get_data__('y_left_wrist_pos_tr')
        return right_pos_traj, left_pos_traj

    def get_wrist_vels_traj(self):
        right_vels_traj = self.__get_data__('y_right_wrist_vels_tr')
        left_vels_traj = self.__get_data__('y_left_wrist_vels_tr')
        return right_vels_traj, left_vels_traj

    def get_local_pos(self):
        return self.__get_data__('y_local_pos')

    def get_local_vel(self):
        return self.__get_data__('y_local_vel')

    def get_next_traj(self):
        rp_tr = self.get_root_pos_traj()
        rv_tr = self.get_root_vel_traj()
        rwp_tr, lwp_tr = self.get_wrist_pos_traj()
        rwv_tr, lwv_tr = self.get_wrist_vels_traj()
        pred_dir = self.get_root_new_forward()

        return rp_tr, rv_tr, rwp_tr, lwp_tr, rwv_tr, lwv_tr, pred_dir

    # def get_local_wrist_pos_curr_fr(self, left_wr_id, right_wrist_id):
