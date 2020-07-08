"""author: Janis Sprenger """
import numpy as np
import math, time

from ...nn.fc_models.fc_networks import FCNetwork
from ..controller import Controller
from .trajectory_box import Trajectory
from .character_box import Character

from ... import utils

DEBUG = False
DEBUG_TIMING = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)


class BoxingController(Controller):
    """
    This is a controller for directional input.

    Returns:
        [type] -- [description]
    """

    def __init__(self, network: FCNetwork, config_store):  # xdim, ydim, end_joints = 5, numJoints = 21):
        self.network = network
        self.xdim = network.input_size
        self.ydim = network.output_size

        # self.precomputed_weights = {}
        # self.precomputed_bins = 50

        self.endJoints = config_store["end_joints"]
        self.n_joints = config_store["numJoints"] + self.endJoints
        self.use_rotations = config_store["use_rotations"]
        self.n_gaits = config_store["n_gaits"]
        self.use_foot_contacts = config_store["use_footcontacts"]
        self.traj_window = config_store["window"]
        self.num_traj_samples = config_store["num_traj_samples"]
        self.traj_step = config_store["traj_step"]
        self.zero_posture = config_store["zero_posture"]
        self.joint_names_ids = config_store["joint_names_ids"]
        self.in_col_names_ids = config_store["col_indices"][0]
        self.out_col_names_ids = config_store["col_indices"][1]

        input_data = np.array([0.0] * self.xdim)
        out_data = np.array([0.0] * self.ydim)
        self.input = MANNInput(input_data, self.n_joints, self.endJoints, self.joint_names_ids, self.in_col_names_ids)
        self.output = MANNOutput(out_data, self.n_joints, self.endJoints, self.use_foot_contacts, self.joint_names_ids,
                                 self.out_col_names_ids)

        self.lastphase = 0

        self.num_coordinate_dims = 3
        self.num_targets = 2  # one for each hand

        self.max_punch_distance = 0.4482340219788639
        self.punch_targets = np.array([0.0] * self.num_coordinate_dims * self.num_targets)

        self.target_vel = np.array((0.0, 0.0, 0.0))
        self.target_dir = np.array((0.0, 0.0, 0.0))

        self.traj = Trajectory(config_store)
        self.char = Character(config_store)
        self.config_store = config_store
        self.__initialize()

    def pre_render(self, punch_phase, punch_targets):
        """
        This function is called before rendering. It prepares character and trajectory to be rendered.

        Arguments:
            punch_targets {np.array(6)} --{left_punch_target,right_punch_target} user input punch target

        Returns:

        """
        if DEBUG:
            print("\n\n############## PRE RENDER ###########")
            print("input punch targets: ", punch_targets, "")
        if DEBUG_TIMING:
            start_time = time.time()

        # 1. Update target dir and vel based on input
        # #TO DO understand
        # # 1.Why there is a factor of 2.5
        # # 2.Why is target:vel being mixed with direction? => direction is a unit vector and velocity is a vector os vel magnitude * dir gives velocity
        # target_vel_speed = 2.5 * np.linalg.norm(direction)
        # # (1-0.9) * self.target_vel + 0.9 * target_vel_speed * direction => More importance to new vel
        # self.target_vel = utils.glm_mix(self.target_vel, target_vel_speed * direction, 0.9)
        #
        # # if velocity is very less, use previous target_dir else calculate new from default
        # vel_magnitude_condition = utils.euclidian_length(self.target_vel) < 1e-05
        # target_vel_dir_default = utils.normalize(self.target_vel)
        # target_vel_dir = self.target_dir if vel_magnitude_condition else target_vel_dir_default
        #
        # self.target_dir = utils.mix_directions(self.target_dir, target_vel_dir, 0.9)
        #
        # if DEBUG:
        #     print("updated target_dir: ", self.target_vel, self.target_dir)

        # 2. Set new punch_phase and new punch target based on input
        right_p_ph = punch_phase[0]
        left_p_ph = punch_phase[1]
        self.input.set_punch_phase(right_p_ph, left_p_ph)

        right_p_target = punch_targets[:self.num_coordinate_dims]
        left_p_target = punch_targets[self.num_coordinate_dims:]
        self.input.set_punch_target(right_p_target, left_p_target)

        # 3. Update/calculate trajectory based on input
        self.traj.compute_future_wrist_trajectory()

        # 3. Set Trajectory input
        # 4. Prepare and Set Joint Input
        # Steps 3 and 4 will update MANN Input completely to the NN's current input
        # 5. Predict: Get MANN Output
        # 6. Process Prediction i.e. apply rotations and set traj foot drifting
        # 7. Set character's new pose

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
        stand_amount = self.traj.step_forward(self.output.getRotVel())

        # 1. update and smooth trajectory
        self.traj.update_from_predict(self.output.getNextTraj())
        if DEBUG:
            print("punch_phase computation: ", stand_amount, self.output.getdDPhase(), self.lastphase, "")

        # 2. update punch_phase
        self.lastphase = (self.lastphase + (stand_amount) * self.output.getdDPhase()) % (1.0)
        if DEBUG_TIMING:
            print("post_predict: %f" % (time.time() - start_time))
        return self.lastphase

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
        root_pos, root_rot = np.array(self.char.root_position), np.array(self.char.root_rotation)
        pos, vel = self.char.getLocalJointPosVel(root_pos, root_rot)
        return np.reshape(pos, (self.char.joints, 3))

    def getWorldPosRot(self):
        return np.array(self.char.root_position), np.array(self.char.root_rotation)

    def __initialize(self):
        # self.output.data = np.array(self.network.norm["Ymean"])  # * self.network.norm["Ystd"] + self.network.norm["Ymean"]
        # self.input.data = np.array(self.network.norm["Xmean"])  # * self.network.norm["Xstd"] + self.network.norm["Xmean"]

        self.input.data = np.array(self.network.norm["Xmean"])
        self.output.data = np.array(self.network.norm["Ymean"])
        # od = self.network.forward_pass(self.input.data)

        ###################################################################
        # 6. Process predictions
        # self.char.root_position, self.char.root_rotation = self.traj.getWorldPosRot()

        if self.use_rotations:
            raise ValueError("Use Rotations not yet supported")
            # joint_rotations = self.output.getRotations()  # twist rotations
        else:
            joint_rotations = []

        joint_positions = self.output.get_local_pos()
        joint_velocities = self.output.get_local_vel()

        # 7. set new character pose
        self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

        # TODO: Check if to apply before or after set_pose
        self.char.root_rotation += self.output.getRootRotationVelocity()  # [-2*pi, +2*pi]
        self.char.root_position += self.output.getRootVelocity()

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
        self.num_coordinate_dims = 3
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
        curr_phase = np.array([right_phase, left_phase])
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

    def get_punch_change_in_phase(self):
        return self.__get_data__('y_punch_dphase')

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
