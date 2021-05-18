"""author: Chirag Bhuvaneshwara """
import numpy as np

from vis.backend.controller import utils


class MANNOutput(object):
    """
    This class is managing the network output. It is depending on the network data model.

    Arguments:
        object {[type_in]} -- [description]

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, data, n_joints, bone_map, y_column_demarcation_ids):
        self.data = data
        self.joints = n_joints
        # 3 dimensional space
        self.num_coordinate_dims = 3
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

    def get_wrist_local_pos(self):
        lp = self.__get_data__('y_local_pos')

        r_wr_start = self.bone_map["RightWrist"] * 3
        r_wr_lp = lp[r_wr_start: r_wr_start + 3]

        l_wr_start = self.bone_map["LeftWrist"] * 3
        l_wr_lp = lp[l_wr_start: l_wr_start + 3]

        return r_wr_lp, l_wr_lp

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
        # rpunch_tr, lpunch_tr = self.get_punch_labels_traj()
        # pred_dir = self.get_root_new_forward()

        return rp_tr, rv_tr, rwp_tr, lwp_tr, rwv_tr, lwv_tr, \
            # rpunch_tr, lpunch_tr
        # , pred_dir

    def set_wrist_pos_tr(self, right_wrist_pos_traj, left_wrist_pos_traj):
        self.__set_data__("y_right_wrist_pos_tr", right_wrist_pos_traj)
        self.__set_data__("y_left_wrist_pos_tr", left_wrist_pos_traj)
