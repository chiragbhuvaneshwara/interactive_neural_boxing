"""author: Chirag Bhuvaneshwara """

# TODO : Ensure that all manipulations in methods or functions are applied to copies of arrays and not the arrays
#  themselves

import numpy as np
from ... import utils


class Character:
    """
    Character class contains information about the simulated character.

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, data_configuration):  # endJoints = 5, numJoints = 21):
        self.endJoints = data_configuration["end_joints"]
        self.joints = data_configuration["num_joints"] + self.endJoints  # 59

        # fields for joint positions and velocities in global space.
        self.joint_positions = np.array([[0.0, 0.0, 0.0]] * self.joints)
        self.joint_velocities = np.array([[0.0, 0.0, 0.0]] * self.joints)
        self.last_joint_positions = np.array(self.joint_positions)

        self.local_joint_positions = np.array(self.joint_positions)
        self.joint_rotations = np.array([0.0] * self.joints)

        # angle of rotation around up axis -> utils.z_angle(new_forward_dir) + old_root_rotation --> [-2*pi, + 2* pi]
        self.root_rotation = 0.0
        # root position projected to ground (y-axis = 0)
        self.root_position = np.array([0.0, 0.0, 0.0])

    def set_pose(self, joint_positions, joint_velocities, joint_rotations, init=False):
        """
        Sets a new pose after prediction.

        Arguments:
            joint_positions {np.array(njoints * 3)} -- predicted root-local joint positions
            joint_velocities {np.array(njoints * 3)} -- predicted root-local joint velocity
            joint_rotations {[type_in]} -- not utilized at the moment

        Keyword Arguments:
            foot_contacts {list} -- binary vector of foot-contacts  (default: {[0, 0, 0, 0]})
        """
        self.joint_rotations = joint_rotations
        # build local transforms from prediction
        self.last_joint_positions = np.array(self.joint_positions)

        for j in range(0, self.joints):
            local_pos = np.array([joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]],
                                 dtype=np.float64).reshape(1, 3)
            pos = self.convert_local_to_global(local_pos, type_in='pos').ravel()

            local_vel = np.array(
                [joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]],
                dtype=np.float64).reshape(1, 3)
            vel = self.convert_local_to_global(local_vel, type_in='vels').ravel()

            if not init:
                # mix positions and velocities.
                self.joint_positions[j] = utils.glm_mix(self.joint_positions[j] + vel, pos, 0.5)
            elif init:
                self.joint_positions[j] = pos

            self.joint_velocities[j] = vel

    def compute_foot_sliding(self, joint_positions, joint_velocities, foot_contacts=[0, 0, 0, 0]):
        def compute_foot_movement(j):
            local_pos = np.array(
                [joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]],
                dtype=np.float64).reshape(3, )
            pos = utils.rot_around_z_3d(local_pos, self.root_rotation) + self.root_position
            local_vel = np.array(
                [joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]],
                dtype=np.float64).reshape(3, )
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

    def get_local_joint_pos_vel(self, prev_root_pos, prev_root_rot):
        joint_pos = np.array([0.0] * (self.joints * 3), dtype=np.float64)
        joint_vel = np.array([0.0] * (self.joints * 3), dtype=np.float64)
        prp = prev_root_pos
        prr = prev_root_rot

        for i in range(0, self.joints):
            # get previous joint position

            curr_joint_pos = self.joint_positions[i]
            curr_joint_pos = curr_joint_pos.reshape(1, len(curr_joint_pos))
            curr_joint_pos = self.convert_global_to_local(curr_joint_pos, prp, prr, type_in='pos')
            joint_pos[i * 3: i * 3 + 3] = curr_joint_pos.ravel()

            curr_joint_vel = self.joint_velocities[i]
            curr_joint_vel = curr_joint_vel.reshape(1, len(curr_joint_vel))
            curr_joint_vel = self.convert_global_to_local(curr_joint_vel, prp, prr, type_in='vels')
            joint_vel[i * 3:i * 3 + 3] = curr_joint_vel.ravel()

        return joint_pos, joint_vel

    def convert_global_to_local(self, arr, root_pos, root_rot, type_in='pos'):

        arr_copy = np.array(arr)
        root_pos_copy = np.array(root_pos)
        root_rot_copy = np.array(root_rot)

        type_arg = type_in.split("_")
        for i in range(len(arr_copy)):
            if type_arg[0] == 'pos':
                if len(type_arg) > 1 and type_arg[1] == "hand":
                    root_pos_copy[1] = 0
                arr_copy[i] -= root_pos_copy

            arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], root_rot_copy, inverse=True)

        return arr_copy

    def convert_local_to_global(self, arr, type_in='pos'):
        arr_copy = np.array(arr)
        root_pos = np.array(self.root_position)
        root_rot = np.array(self.root_rotation)
        type_arg = type_in.split("_")

        for i in range(len(arr_copy)):
            arr_copy[i] = utils.rot_around_z_3d(arr_copy[i], root_rot)
            if type_arg[0] == 'pos':
                if len(type_arg) > 1 and type_arg[1] == "hand":
                    root_pos[1] = 0
                arr_copy[i] = arr_copy[i] + root_pos
        return arr_copy

    def reset(self, root_position=np.array([0.0, 0.0, 0.0]), start_orientation=0.0):
        self.root_position = root_position
        self.root_rotation = start_orientation
