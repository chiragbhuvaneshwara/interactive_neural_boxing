"""author: Chirag Bhuvaneshwara """


class MANNInput(object):
    """
    This class is managing the network input. It is depending on the network data model

    Arguments:
        object {[type_in]} -- [description]

    Returns:
        [type_in] -- [description]
    """

    def __init__(self, data, n_joints, x_column_demarcation_ids):
        self.data = data
        self.joints = n_joints
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
        return {'right': ph_r, 'left': ph_l}
