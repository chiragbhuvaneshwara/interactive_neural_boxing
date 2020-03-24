def get_punch_targets(self, punch_phase):
    """
        reach_labels: np.array(n_frame)
        :return target_positions (np.array(n_frame, 3))
    """
    # reach_labels, reach_delta = self.load_phase(reach_file)
    # access to global positions
    # self.__global_positions

    # create punch_phase target array
    hand_joint = 16
    punch_target_array = np.array(self.__global_positions[:, hand_joint])

    # todo: iterate through all the labels and get the punch labels in the punch_target_array

    next_target = (0.0, 0.0, 0.0)
    target_pos = []
    tracker = 0

    for i in range(len(punch_phase)):
        if punch_phase[i] == 0.0:
            # Does setting target as (0,0,0) help with punches? Coz the hands are always held up in front of the
            # face
            next_target = np.array([0.0, 0.0, 0.0])

        elif punch_phase[i] == 1.0:
            next_target = punch_target_array[i]

        elif 0.0 < punch_phase[i] < 1.0:
            curr_dir = punch_phase[i] - tracker

            # if np.all(next_target == 0.0) and curr_dir > 0:
            if curr_dir > 0:
                for j in range(i, len(punch_phase)):
                    if punch_phase[j] == 1.0:
                        next_target = punch_target_array[j]
                        break

            else:  # curr_dir < 0
                next_target = np.array([0.0, 0.0, 0.0])

        tracker = punch_phase[i]

        target_pos.append(next_target)

    punch_target_array = np.array(target_pos)

    # normalize the position
    # root_positions = self.__global_positions[:, 0:1, 0], self.__global_positions[:, 0:1, 2] # [idx_1, idx_2, idx_3]

    punch_target_array[:, 0] = punch_target_array[:, 0] - self.__global_positions[:, 0, 0]
    punch_target_array[:, 2] = punch_target_array[:, 2] - self.__global_positions[:, 0, 2]
    root_rotations = self.get_root_rotations()

    for f in range(self.n_frames - 1):
        punch_target_array[f] = root_rotations[f] * punch_target_array[f]

    punch_target_array[punch_phase == 0.0] = [0.0, 0.0, 0.0]

    return punch_target_array