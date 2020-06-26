"""author: Janis Sprenger """
import numpy as np
import math, time

from ..nn.fc_models.fc_networks import FCNetwork
from .controller import Controller
from .trajectory import Trajectory
from .character import Character

from .. import utils

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

    def __init__(self, network: FCNetwork, config_store):  # xdim, ydim, endJoints = 5, numJoints = 21):
        self.network = network

        self.initial = True
        self.xdim = network.input_size  # xdim # 222 # 438 #450
        self.ydim = network.output_size  # ydim #217 # 559
        # self.hdim = 512

        self.precomputed_weights = {}
        self.precomputed_bins = 50

        # self.endJoints = 5
        # self.joints = 21 + self.endJoints # 59
        self.endJoints = config_store["endJoints"]
        self.n_joints = config_store["numJoints"] + self.endJoints
        self.n_gaits = config_store["n_gaits"]
        self.use_rotations = config_store["use_rotations"]
        self.use_foot_contacts = config_store["use_footcontacts"]
        self.zero_posture = config_store["zero_posture"]

        input_data = np.array([0.0] * self.xdim)
        out_data = np.array([0.0] * self.ydim)
        self.input = MANNInput(input_data, self.n_joints, self.n_gaits, self.endJoints, self.use_foot_contacts)
        self.output = MANNOutput(out_data, self.n_joints, self.endJoints, self.use_foot_contacts)

        self.lastphase = 0

        num_coordinate_dims = 3
        num_targets = 2  # one for each hand

        self.max_punch_distance = 0.4482340219788639
        self.punch_targets = np.array([0.0] * num_coordinate_dims * num_targets)

        # self.target_vel = np.array((0.0, 0.0, 0.0))
        # self.target_dir = np.array((0.0, 0.0, 0.0))

        # self.traj = Trajectory()
        self.char = Character(config_store)
        self.config_store = config_store
        self.__initialize()

    def pre_render(self, punch_targets):
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

        # 1. Compute Punch Phase for both hands w.r.t maximum punch distance from dataset
        curr_phase = self.char.compute_punch_phase(punch_targets)

        # 2. Set new phase as input
        self.input.set_punch_phase(curr_phase)
        print(curr_phase)

        # 3. Set new punch targets (or blend old and new punch targets)
        self.input.set_punch_target(punch_targets)

        # 4. Prepare and set joint input
        # prev_root_pos, prev_root_rot = self.traj.getPreviousPosRot()
        prev_root_pos = self.char.root_position
        prev_root_rot = self.char.root_rotation
        joint_pos, joint_vel = self.char.getLocalJointPosVel(prev_root_pos, prev_root_rot)
        self.input.set_local_pos(joint_pos)
        self.input.set_local_vel(joint_vel)

        # 5. Make predictions
        if DEBUG_TIMING:
            pre_predict = time.time()
        self.output.data = self.network.forward_pass(self.input.data)
        if DEBUG_TIMING:
            post_predict = time.time()
        ###############################################
        # 6. Process predictions
        # self.char.root_position, self.char.root_rotation = self.traj.getWorldPosRot()

        if self.use_rotations:
            joint_rotations = self.output.getRotations()  # twist rotations
        else:
            joint_rotations = []

        joint_positions = self.output.getJointPos()
        joint_velocities = self.output.getJointVel()

        # 7. set new character pose
        self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

        # TODO: Check if to apply before or after set_pose
        self.char.root_rotation += self.output.getRootRotationVelocity()  # [-2*pi, +2*pi]
        self.char.root_position += self.output.getRootVelocity()
        # self.char.root_position = self.output.getRotVel()[0]

        if DEBUG_TIMING:
            print("prerender: %f, from this predict: %f" % (time.time() - start_time, post_predict - pre_predict))

        # return self.char.root_position

    def post_render(self):
        """
        This function has to be called after rendering to prepare the next frame.

        Returns:
            float -- changed phase depending on user output.
        """
        if DEBUG:
            print("\n\n############## POST RENDER ###########")
        if DEBUG_TIMING:
            start_time = time.time()
        stand_amount = self.traj.step_forward(self.output.getRotVel())

        # 1. update and smooth trajectory
        self.traj.update_from_predict(self.output.getNextTraj())
        if DEBUG:
            print("phase computation: ", stand_amount, self.output.getdDPhase(), self.lastphase, "")

        # 2. update phase
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
        # self.set_weights(n.W0, n.W1, n.W2, n.b0, n.b1, n.b2, n.xmean, n.xstd, n.ymean, n.ystd)
        # self.network = n

        # if self.initial or True:
        self.output.data = np.array(
            self.network.norm["Ymean"])  # * self.network.norm["Ystd"] + self.network.norm["Ymean"]
        self.input.data = np.array(
            self.network.norm["Xmean"])  # * self.network.norm["Xstd"] + self.network.norm["Xmean"]

        od = self.network.forward_pass(self.input.data)

        ###################################################################
        # 6. Process predictions
        # self.char.root_position, self.char.root_rotation = self.traj.getWorldPosRot()

        if self.use_rotations:
            joint_rotations = self.output.getRotations()  # twist rotations
        else:
            joint_rotations = []

        joint_positions = self.output.getJointPos()
        joint_velocities = self.output.getJointVel()

        # 7. set new character pose
        self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

        # TODO: Check if to apply before or after set_pose
        self.char.root_rotation += self.output.getRootRotationVelocity()  # [-2*pi, +2*pi]
        self.char.root_position += self.output.getRootVelocity()

    # self.char.root_position = self.output.getRotVel()[0]

    ###########################################################################################

    # def getEndJointRotations(self):
    # 	return self.out[self.out_end_joint_rot_base:self.out_end_joint_rot_base + self.endJoints * 3]

    def __update_target_dir_simple(self, direction):
        # Compute target direction from
        target_vel_speed = 2.5  # 0.05												# target velocity factor, has to be adapted to dataset!
        self.target_vel = direction * target_vel_speed
        self.target_dir = utils.normalize(np.array(self.target_vel))


class MANNInput(object):
    """
    This class is managing the network input. It is depending on the network data model

    Arguments:
        object {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, data, joints, n_gaits, endJoints, use_foot_contacts):
        self.data = data
        self.joints = joints
        # Since we are working in 3 dimensional space
        self.num_coordinate_dims = 3

        # Variables marking the start positions of the columns in the input
        # punch target is a vector of size 3 occupying 3 columns
        self.in_root_base = 0
        # self.in_punch_phase_left = self.in_root_base
        # self.in_punch_phase_right = self.in_punch_phase_left + 1
        #
        # self.in_punch_target_left = self.in_punch_phase_right + 1
        # self.in_punch_target_right = self.in_punch_target_left + 1 * self.num_coordinate_dims
        #
        # self.in_local_pos = self.in_punch_target_right + 1 * self.num_coordinate_dims

        self.in_punch_phase_right = self.in_root_base
        self.in_punch_phase_left = self.in_punch_phase_right + 1

        self.in_punch_target_right = self.in_punch_phase_left + 1
        self.in_punch_target_left = self.in_punch_target_right + 1 * self.num_coordinate_dims

        self.in_local_pos = self.in_punch_target_left + 1 * self.num_coordinate_dims

        self.in_local_vel = self.in_local_pos + self.num_coordinate_dims * self.joints

    # self.n_gaits = n_gaits
    # self.in_traj_pos_base = 0
    # self.in_traj_dir_base = self.in_traj_pos_base + 2 * 12
    # self.in_traj_gait_base = self.in_traj_dir_base + 2 * 12
    # self.in_joint_pos = self.in_traj_gait_base + self.n_gaits * 12
    # self.in_joint_vel = self.in_joint_pos + 3 * self.joints
    # self.in_joint_twist = self.in_joint_vel + 3 * self.joints
    # self.endJoints = endJoints

    # return

    def set_punch_phase(self, curr_phase):
        self.data[self.in_punch_phase_left] = curr_phase[1]
        self.data[self.in_punch_phase_right] = curr_phase[0]

    def set_punch_target(self, targets):
        # self.data[self.in_punch_target_left: self.in_punch_target_left + self.num_coordinate_dims] = targets[
        #                                                                                              :self.num_coordinate_dims]
        self.data[self.in_punch_target_left: self.in_punch_target_left + self.num_coordinate_dims] = targets[
                                                                                                     self.num_coordinate_dims:]
        # self.data[self.in_punch_phase_right: self.in_punch_phase_right + self.num_coordinate_dims] = targets[
        #                                                                                              self.num_coordinate_dims:]
        self.data[self.in_punch_phase_right: self.in_punch_phase_right + self.num_coordinate_dims] = targets[
                                                                                                     :self.num_coordinate_dims]

    def set_local_pos(self, pos):
        self.data[self.in_local_pos: self.in_local_pos + self.num_coordinate_dims * self.joints] = pos[:]

    def set_local_vel(self, vel):
        self.data[self.in_local_vel:self.in_local_vel + self.num_coordinate_dims * self.joints] = vel[:]


# def setJointTwist(self, twist):
#     self.data[self.in_joint_twist:self.in_joint_twist + self.joints - self.endJoints] = np.reshape(twist, [
#         self.joints - self.endJoints, ])
#
# def setTrajPos(self, pos):
#     self.data[self.in_traj_pos_base:self.in_traj_pos_base + 2 * 12] = pos[:]  # np.reshape(pos, [24])
#
# def getInputTrajPos(self):
#     return self.data[self.in_traj_pos_base:self.in_traj_pos_base + 2 * 12]
#
# def setTrajDir(self, dirs):
#     self.data[self.in_traj_dir_base:self.in_traj_dir_base + 2 * 12] = dirs[:]  # np.reshape(dirs, [24])
#
# def setTrajGait(self, gaits):
#     self.data[self.in_traj_gait_base:self.in_traj_gait_base + self.n_gaits * 12] = gaits[
#                                                                                    :]  # np.reshape(gaits,[gaits.size])
#
# def setJointPos(self, pos):
#     self.data[self.in_joint_pos:self.in_joint_pos + 3 * self.joints] = pos[:]  # np.reshape(pos, [pos.size, 1])
#
# def setJointVel(self, vel):
#     self.data[self.in_joint_vel:self.in_joint_vel + 3 * self.joints] = vel  # np.reshape(vel, [vel.size, 1])
#
# def getLastTrajPos(self):
#     return np.array([self.data[self.in_traj_pos_base + 1:self.in_traj_pos_base + 7],
#                      self.data[self.in_traj_pos_base + 13:self.in_traj_pos_base + 13 + 6]])
#
# def getLastTrajDir(self):
#     return np.array([self.data[self.in_traj_dir_base + 1:self.in_traj_dir_base + 7],
#                      self.data[self.in_traj_dir_base + 13:self.in_traj_dir_base + 13 + 6]])
#
# def getLastTrajGait(self):
#     return self.data[self.in_traj_gait_base + self.n_gaits:self.in_traj_gait_base + self.n_gaits * 7]


class MANNOutput(object):
    """
    This class is managing the network output. It is depending on the network data model.

    Arguments:
        object {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, data, joints, endJoints, use_foot_contacts):
        self.data = data
        self.joints = joints
        # Since we are working in 3 dimensional space
        self.num_coordinate_dims = 3
        self.out_root_base = 0

        self.out_root_vel_X = self.out_root_base
        self.out_root_vel_Y = self.out_root_vel_X + 1

        self.out_root_new_forward = self.out_root_vel_Y + 1  # 2D
        self.out_root_foot_l = self.out_root_new_forward + 2  # 2D
        self.out_root_foot_r = self.out_root_foot_l + 2  # 2D

        self.out_joint_pos_base = self.out_root_foot_r + 2
        self.out_joint_vel_base = self.out_joint_pos_base + self.joints * self.num_coordinate_dims

        self.endJoints = endJoints

    # self.out_dphase_base = self.out_root_base + 4
    # self.out_contacts_base = self.out_dphase_base + 1
    # if use_foot_contacts:
    # 	self.out_next_traj_base = self.out_root_new_forward + 2
    # else:
    # 	self.out_next_traj_base = self.out_contacts_base
    # self.out_joint_pos_base = self.out_next_traj_base + 2 * 2 * 6
    # self.out_joint_vel_base = self.out_joint_pos_base + self.joints * 3
    # self.out_joint_rot_base = self.out_joint_vel_base + self.joints * 3
    # # self.out_end_joint_rot_base = self.out_joint_rot_base + self.joints
    # self.endJoints = endJoints
    # return

    def getFootContacts(self):
        return np.array(self.data[self.out_root_foot_l:self.out_root_foot_r + 2])

    def getRootVelocity(self):
        return np.array([self.data[self.out_root_vel_X], 0, self.data[self.out_root_vel_Y]])

    def getRootRotationVelocity(self):
        return utils.z_angle(
            np.array([self.data[self.out_root_new_forward], 0, self.data[self.out_root_new_forward + 1]]))

    # def getRotVel(self):
    # return self.data[self.out_root_base:self.out_root_base + 3]
    # return np.array(self.data[self.out_root_base:self.out_root_base + 4])

    # def getdDPhase(self):
    # 	return np.array(self.data[self.out_dphase_base])

    # def getNextTraj(self):
    # 	return np.array(self.data[self.out_next_traj_base:self.out_next_traj_base + (2 * 2 * 6)])

    def getJointPos(self):
        arr = np.array(
            self.data[self.out_joint_pos_base: self.out_joint_pos_base + (self.num_coordinate_dims * self.joints)])
        arr = arr.reshape((self.joints, self.num_coordinate_dims))
        return arr

    def getJointVel(self):
        arr = np.array(
            self.data[self.out_joint_vel_base:self.out_joint_vel_base + self.num_coordinate_dims * self.joints])
        arr = arr.reshape((self.joints, self.num_coordinate_dims))

        return arr

# def getRotations(self):
# 	return np.array(self.data[self.out_joint_rot_base:self.out_joint_rot_base + 1 * (self.joints - self.endJoints)])
