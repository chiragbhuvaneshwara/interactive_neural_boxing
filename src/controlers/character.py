"""author: Janis Sprenger """
import numpy as np
import math, time

from ..nn.fc_models.fc_networks import FCNetwork

from .. import utils

DEBUG = False
DEBUG_TIMING = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)


class Character:
    """
    Character class contains information about the simulated character.

    Returns:
        [type] -- [description]
    """

    def __init__(self, config_store):  # endJoints = 5, numJoints = 21):
        self.endJoints = config_store["endJoints"]
        self.joints = config_store["numJoints"] + self.endJoints  # 59

        # fields for joint positions and velocities in global space.
        self.joint_positions = np.array([[0.0, 0.0, 0.0]] * self.joints)
        self.joint_velocities = np.array([[0.0, 0.0, 0.0]] * self.joints)
        self.last_joint_positions = np.array(self.joint_positions)

        # Indices from dataset_config json
        # Janis says indices from dataset_config
        self.foot_left = [4, 5]
        self.foot_right = [9, 10]

        self.shoulder_left = 36
        self.hand_left = 39
        self.shoulder_right = 13
        self.hand_right = 16

        ########################################################################################################################
        self.max_punch_distance = 0.4482340219788639
        ########################################################################################################################

        self.local_joint_positions = np.array(self.joint_positions)
        self.joint_rotations = np.array([0.0] * self.joints)

        # angle of rotation around up axis -> utils.z_angle(new_forward_dir) + old_root_rotation --> [-2*pi, + 2* pi]
        self.root_rotation = 0.0
        self.running = 0.0
        # root position projected to ground (y-axis = 0)
        self.root_position = np.array([0.0, 0.0, 0.0])

    def reset(self, root_position, start_orientation):
        self.root_position = root_position
        self.root_rotation = start_orientation
        self.running = 0.0

    def set_pose(self, joint_positions, joint_velocities, joint_rotations, foot_contacts=[0, 0, 0, 0]):
        """
        Sets a new pose after prediction.

        Arguments:
            joint_positions {np.array(njoints * 3)} -- predicted root-local joint positions
            joint_velocities {np.array(njoints * 3)} -- predicted root-local joint velocity
            joint_rotations {[type]} -- not utilized at the moment

        Keyword Arguments:
            foot_contacts {list} -- binary vector of foot-contacts  (default: {[0, 0, 0, 0]})
        """
        self.joint_rotations = joint_rotations
        # build local transforms from prediction
        self.last_joint_positions = np.array(self.joint_positions)

        for j in range(0, self.joints):
            # local_pos = np.array(
            #     [joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]]).reshape(3, )
            local_pos = np.array(joint_positions[j]).reshape(3, )
            pos = utils.rot_around_z_3d(local_pos, self.root_rotation) + self.root_position
            # local_vel = np.array(
            #     [joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]]).reshape(3, )
            local_vel = np.array(joint_velocities[j]).reshape(3, )
            vel = utils.rot_around_z_3d(local_vel, self.root_rotation)

            self.joint_positions[j] = utils.glm_mix(self.joint_positions[j] + vel, pos,
                                                    0.5)  # mix positions and velocities.
            # self.local_joint_positions[j] = utils.rot_around_z_3d(self.joint_positions[j] - self.root_position, self.root_rotation, inverse = True)
            self.joint_velocities[j] = vel
        # prediction is finished and post processed. Pose can be rendered!
        return

    def compute_punch_phase(self, punch_targets):

        joint_positions = self.joint_positions

        def compute_global_positions(j):
            # j = self.hand_left
            # joint_positions
            local_pos = np.array(joint_positions[j]).reshape(3, )
            pos = utils.rot_around_z_3d(local_pos, self.root_rotation) + self.root_position
            return pos

        def compute_arm_dist(i, j):
            pos_shoulder = compute_global_positions(j)
            pos_hand = compute_global_positions(i)
            arm_distance = np.linalg.norm(pos_shoulder - pos_hand)
            return arm_distance

        # def get_acting_arm():
        #     l_dist = compute_arm_dist(self.hand_left, self.shoulder_left)
        #     r_dist = compute_arm_dist(self.hand_right, self.shoulder_right)
        #
        #     if l_dist > r_dist:
        #         return 'left', l_dist
        #
        #     else:
        #         return 'right', r_dist

        def get_acting_arm(punch_targets):
            l_dist = compute_arm_dist(self.hand_left, self.shoulder_left)
            r_dist = compute_arm_dist(self.hand_right, self.shoulder_right)

            l_target = punch_targets[0:3]
            r_target = punch_targets[3:]
            no_movement_target = np.array([0.0, 0.0, 0.0])

            l_target_dist = np.linalg.norm(l_target - no_movement_target)
            r_target_dist = np.linalg.norm(r_target - no_movement_target)

            if l_target_dist > r_target_dist:
                return 'left', l_dist

            else:
                return 'right', r_dist

        def convert_range(new_max, new_min, old_max, old_min, old_value):
            old_range = (old_max - old_min)
            new_range = (new_max - new_min)
            new_value = (((old_value - old_min) * new_range) / old_range) + new_min
            return new_value

        acting_arm, dist = get_acting_arm(punch_targets)
        acting_arm_phase = convert_range(1, 0, self.max_punch_distance, 0, dist)

        if acting_arm == 'left':
            return np.array([acting_arm_phase, 0])
        else:
            return np.array([0, acting_arm_phase])

    def compute_foot_sliding(self, joint_positions, joint_velocities, foot_contacts=[0, 0, 0, 0]):
        def compute_foot_movement(j):
            local_pos = np.array(
                [joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]]).reshape(3, )
            pos = utils.rot_around_z_3d(local_pos, self.root_rotation) + self.root_position
            local_vel = np.array(
                [joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]]).reshape(3, )
            vel = utils.rot_around_z_3d(local_vel, self.root_rotation)
            return self.joint_positions[j] - utils.glm_mix(self.joint_positions[j] + vel, pos, 0.5)

        global_foot_drift = np.array([0.0, 0.0, 0.0])
        if foot_contacts[0]:
            global_foot_drift += compute_foot_movement(self.foot_left[0])
        if foot_contacts[1]:
            global_foot_drift += compute_foot_movement(self.foot_left[1])
        if foot_contacts[2]:
            global_foot_drift += compute_foot_movement(self.foot_right[0])
        if foot_contacts[3]:
            global_foot_drift += compute_foot_movement(self.foot_right[1])

        if (np.sum(foot_contacts) > 0):
            global_foot_drift /= np.sum(foot_contacts)
            global_foot_drift[1] = 0.0
        # print("foot correction: ", foot_contacts, global_foot_drift)
        # return np.array([0.0, 0.0, 0.0])
        return global_foot_drift

    def getLocalJointPosVel(self, prev_root_pos, prev_root_rot):
        joint_pos = np.array([0.0] * (self.joints * 3))
        joint_vel = np.array([0.0] * (self.joints * 3))

        for i in range(0, self.joints):
            # get previous joint position

            # pos = utils.rot_around_z_3d(self.char.joint_positions[i] - prev_root_pos, prev_root_rot, inverse=True)  # self.char.joint_positions[i]#
            pos = utils.rot_around_z_3d(self.joint_positions[i] - prev_root_pos,
                                        -prev_root_rot)  # self.char.joint_positions[i]#
            joint_pos[i * 3 + 0] = pos[0]
            joint_pos[i * 3 + 1] = pos[1]
            joint_pos[i * 3 + 2] = pos[2]

            # get previous joint velocity
            # vel = utils.rot_around_z_3d(self.char.joint_velocities[i], prev_root_rot,inverse=True)  # self.char.joint_velocities[i] #
            vel = utils.rot_around_z_3d(self.joint_velocities[i], -prev_root_rot)  # self.char.joint_velocities[i] #
            joint_vel[i * 3 + 0] = vel[0]
            joint_vel[i * 3 + 1] = vel[1]
            joint_vel[i * 3 + 2] = vel[2]

        return (joint_pos, joint_vel)
