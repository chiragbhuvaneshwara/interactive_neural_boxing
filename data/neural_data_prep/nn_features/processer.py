import os
import numpy as np

from common.utils import retrieve_name

from data.neural_data_prep.nn_features.extractor import FeatureExtractor
from data.raw_data.punch_label_gen.analyze.stats import gen_punch_stats


def get_files_in_folder(folder):
    """
    :@param folder: str, path to a folder
    :@return:
    files: list, sorted list of files in folder
    """

    files = []
    dir_files = os.listdir(folder)
    dir_files.sort()
    for path in dir_files:
        full_path = os.path.join(folder, path)
        files.append(full_path)

    return files


def prepare_col_demarcation_ids(**args):
    """
    :@param args: all the variables that can be possibly used as part of X or Y
    :@return:
    col_demarcation_ids: dict of joint names with start_idx and end_idx
    col_names: list of col names with for example position marked as position_0, position_1, position_2 for x,y,z co-ordinates
    """
    keys = args.keys()
    col_names = []

    start = 0
    end = 0
    col_demarcation_ids = {}
    for k, v in zip(keys, args.values()):
        end = len(v) + end
        col_demarcation_ids[k] = [start, end]
        col_names_subset = [k] * (end - start)
        col_names_subset = [var + '_' + str(i) for i, var in enumerate(col_names_subset)]
        col_names.extend(col_names_subset)
        start = end

    print('\n')
    print('col names len: ', len(col_names))

    return col_demarcation_ids, col_names


def prepare_input_data(frame_num, handler, col_demarcation_finished=True):
    """
    Prepares one datapoint of the neural network input data.

    @param frame_num: int, frame number of current bvh data
    @param handler: FeatureExtractor instance, corresponding to current bvh and punch label files
    @param col_demarcation_finished: bool, flag to denote whether to print the col_demarcation info or not
    @return:
    x_curr_frame: list, one datapoint of the neural network input data
    """
    # TODO Add docstring after simplifying inputs i.e after moving all these input vars inside the handler

    punch_labels = handler.punch_labels_binary
    punch_target = handler.punch_targets
    local_positions = handler.get_root_local_joint_positions()
    local_velocities = handler.get_root_local_joint_velocities()

    x_root_fwd = handler.get_forward_directions().flatten()

    traj_info = handler.get_trajectory(frame_num)

    x_root_pos_tr = traj_info['root_pos']
    x_root_pos_tr = np.delete(x_root_pos_tr, 1, 1).ravel()  # Trajectory Pos, 2 * 10d

    x_root_vels_tr = traj_info['root_vels']
    x_root_vels_tr = np.delete(x_root_vels_tr, 1, 1).ravel()

    x_root_dirs_tr = traj_info['root_dirs']
    x_root_dirs_tr = np.delete(x_root_dirs_tr, 1, 1).ravel()

    x_right_wrist_pos_tr = traj_info['right_wrist_pos'].ravel()
    x_left_wrist_pos_tr = traj_info['left_wrist_pos'].ravel()

    x_right_wrist_vels_tr = traj_info['right_wrist_vels'].ravel()
    x_left_wrist_vels_tr = traj_info['left_wrist_vels'].ravel()

    x_right_punch_labels_tr = traj_info['right_punch_labels'].ravel()
    x_left_punch_labels_tr = traj_info['left_punch_labels'].ravel()

    x_right_punch_labels = punch_labels[handler.hand_right][frame_num].ravel()
    x_left_punch_labels = punch_labels[handler.hand_left][frame_num].ravel()

    x_right_punch_target = punch_target[handler.hand_right][frame_num].ravel()
    x_left_punch_target = punch_target[handler.hand_left][frame_num].ravel()

    x_local_pos = local_positions[frame_num - 1].ravel()
    x_local_vel = local_velocities[frame_num - 1].ravel()

    x_curr_frame = [
        x_root_pos_tr,  # local wrt r in mid frame
        x_root_vels_tr,
        x_root_dirs_tr,
        x_right_wrist_pos_tr,  # local wrt r in mid frame (TODO then wrt wrist in mid frame)
        x_left_wrist_pos_tr,  # local wrt r in mid frame (TODO then wrt wrist in mid frame)
        x_right_wrist_vels_tr,
        x_left_wrist_vels_tr,
        # x_right_punch_labels_tr,
        # x_left_punch_labels_tr,
        x_right_punch_labels,  # TODO:Janis said remove
        x_left_punch_labels,  # TODO:Janis said remove
        x_right_punch_target,  # local wrt r in mid frame                                   #TODO:Janis said remove
        x_left_punch_target,  # local wrt r in mid frame                                   #TODO:Janis said remove
        x_local_pos,
        x_local_vel
    ]

    if not col_demarcation_finished:
        keys = list(map(retrieve_name, x_curr_frame))
        kwargs = {k: v for k, v in zip(keys, x_curr_frame)}
        x_demarcation_ids, x_col_names = prepare_col_demarcation_ids(**kwargs)
        print('curr frame len ', len(np.hstack(x_curr_frame)))
        # print(x_demarcation_ids)
        print('\n')
        return x_demarcation_ids, x_col_names
    else:
        return x_curr_frame


def prepare_output_data(frame_num, handler, col_demarcation_finished=True):
    """
    Prepares one datapoint of the neural network output data.

    @param frame_num: int, frame number of current bvh data
    @param handler: FeatureExtractor instance, corresponding to current bvh and punch label files
    @param col_demarcation_finished: bool, flag to denote whether to print the col_demarcation info or not
    @return:
    y_curr_frame: list, one datapoint of the neural network output data
    """

    punch_labels = handler.punch_labels_binary
    punch_target = handler.punch_targets
    local_positions = handler.get_root_local_joint_positions()
    local_velocities = handler.get_root_local_joint_velocities()
    root_new_forward = handler.new_fwd_dirs
    root_velocity = handler.get_root_velocity()

    traj_info_next = handler.get_trajectory(frame_num + 1, frame_num + 1)

    y_root_pos_tr = traj_info_next['root_pos']
    y_root_pos_tr = np.delete(y_root_pos_tr, 1, 1).ravel()

    y_root_vels_tr = traj_info_next['root_vels']
    y_root_vels_tr = np.delete(y_root_vels_tr, 1, 1).ravel()

    y_right_wrist_pos_tr = traj_info_next['right_wrist_pos'].ravel()
    y_left_wrist_pos_tr = traj_info_next['left_wrist_pos'].ravel()

    y_right_wrist_vels_tr = traj_info_next['right_wrist_vels'].ravel()
    y_left_wrist_vels_tr = traj_info_next['left_wrist_vels'].ravel()

    y_right_punch_labels_tr = traj_info_next['right_punch_labels'].ravel()
    y_left_punch_labels_tr = traj_info_next['left_punch_labels'].ravel()

    y_right_punch_labels = punch_labels[handler.hand_right][frame_num + 1].ravel()
    y_left_punch_labels = punch_labels[handler.hand_left][frame_num + 1].ravel()

    y_right_punch_target = punch_target[handler.hand_right][frame_num + 1].ravel()
    y_left_punch_target = punch_target[handler.hand_right][frame_num + 1].ravel()

    y_local_pos = local_positions[frame_num].ravel()
    y_local_vel = local_velocities[frame_num].ravel()

    y_root_velocity = np.hstack([root_velocity[frame_num, 0, 0].ravel(), root_velocity[frame_num, 0, 2].ravel()])
    y_root_new_forward = root_new_forward[frame_num].ravel()

    # Taking i because length of phase is l and length of dphase is l-1
    # y_punch_dphase = punch_dphase[i].ravel()

    # TODO You have changed foot contacts from l, r to r, l. Ensure that controller can process r, l
    y_right_foot_contacts = handler.foot_contacts[handler.foot_right['a']][frame_num]
    y_left_foot_contacts = handler.foot_contacts[handler.foot_left['a']][frame_num]
    # y_foot_contacts = np.concatenate([feet_r[i], feet_l[i]], axis=-1)

    y_curr_frame = [
        y_root_velocity,
        y_root_new_forward,
        y_root_pos_tr,
        y_root_vels_tr,
        y_right_punch_labels,  # TODO:Janis said remove
        y_left_punch_labels,  # TODO:Janis said remove
        y_right_wrist_pos_tr,
        y_left_wrist_pos_tr,
        y_right_wrist_vels_tr,
        y_left_wrist_vels_tr,
        # y_right_punch_labels_tr,                                   #TODO:Janis said remove
        # y_left_punch_labels_tr,                                   #TODO:Janis said remove
        y_right_foot_contacts,
        y_left_foot_contacts,
        y_local_pos,
        y_local_vel
    ]

    if not col_demarcation_finished:
        keys = list(map(retrieve_name, y_curr_frame))
        kwargs = {k: v for k, v in zip(keys, y_curr_frame)}
        y_demarcation_ids, y_col_names = prepare_col_demarcation_ids(**kwargs)
        print('curr frame len ', len(np.hstack(y_curr_frame)))
        # print(y_demarcation_ids)
        print('\n')
        return y_demarcation_ids, y_col_names
    else:
        return y_curr_frame


def process_data(handler: FeatureExtractor, punch_labels_csv_path, frame_rate_div, forward_dir, tr_window_root,
                 tr_window_wrist, develop,
                 gen_data_config=False, punch_stats=None):
    """
    Generates and stacks the X and Y data for one provided bvh file and punch labels file.
    :@param handler: FeatureExtractor,
    :@param punch_p_csv_path: str,
    :@param frame_rate_div: int, # if 2, Reduces fps from 120fps to 60fps (60 fps reqd. for Axis Neuron bvh)
    :@param develop: boolean, if True processes only 50 frames of mocap data
    :@param gen_data_config: boolean, if True generates the dataset config, else dataset config returned is empty
    :@return:
     x: np.array(n_frames_file, n_x_cols)
     y: np.array(n_frames_file, n_y_cols)
     dataset_config: dict, containing parameters used to generate dataset and some info for accessing different
     variables in input and output vectors.
    """
    x, y = [], []

    for div in range(frame_rate_div):
        print('\n')
        print('Processing Blender csv data %s' % frame_rate_div, div)
        handler.set_awinda_parameters()  # set the skeleton parameters by manually checking the bvh files
        # Loading Bvh file into memory
        handler.load_motion(frame_rate_divisor=frame_rate_div, frame_rate_offset=div)
        handler.load_punch_action_labels(punch_labels_csv_path, frame_rate_divisor=frame_rate_div,
                                         frame_rate_offset=div)

        # TODO Only implemented for action label type tertiary currently. Must do binary labels and phase.
        handler.calculate_punch_targets(space="local")
        handler.calculate_new_forward_dirs()
        handler.get_foot_concats()

        #TODO: Make sure larger among handler.window_root and window_wrist is used both below and in for loop
        x_col_demarcation_ids, x_col_names = prepare_input_data(handler.window_root, handler,
                                                                col_demarcation_finished=False)
        y_col_demarcation_ids, y_col_names = prepare_output_data(handler.window_root, handler,
                                                                 col_demarcation_finished=False)

        for i in range(handler.window_root, handler.n_frames - handler.window_root - 1, 1):
            if i % 50 == 0:
                print('Frames processed: ', i)
                if develop:
                    break
            x_curr_frame = prepare_input_data(i, handler)
            x.append(np.hstack(x_curr_frame))

            y_curr_frame = prepare_output_data(i, handler)
            y.append(np.hstack(y_curr_frame))

    if gen_data_config:
        dataset_config = {
            "left_wrist_pos_avg_diff": handler.left_wrist_pos_avg_diff_punch.tolist(),
            "right_wrist_pos_avg_diff": handler.right_wrist_pos_avg_diff_punch.tolist(),

            "left_wrist_no_punch": {k: v.tolist() for k, v in handler.left_wrist_pos_no_punch.items()},
            "right_wrist_no_punch": {k: v.tolist() for k, v in handler.right_wrist_pos_no_punch.items()},

            "frame_rate_div": frame_rate_div,
            "forward_dir": list(forward_dir),
            "traj_window_root": tr_window_root,
            "traj_window_wrist": tr_window_wrist,

            "num_traj_samples": handler.num_traj_sampling_pts,
            "traj_step_root": handler.traj_step_root,
            "traj_step_wrist": handler.traj_step_wrist,
            "num_joints": len(handler.joint_id_map.keys()),
            "zero_posture": handler.reference_skeleton,
            "bone_map": handler.joint_id_map,

            "col_demarcation_ids": [x_col_demarcation_ids, y_col_demarcation_ids],
            "col_names": [x_col_names, y_col_names],

            "bvh_path": os.path.split(handler.bvh_file_path)[0],
            "punch_label_path": os.path.split(punch_labels_csv_path)[0],
            "punch_stats": punch_stats,
            "vars_input": list(x_col_demarcation_ids.keys()),
            "vars_output": list(y_col_demarcation_ids.keys()),
            "in_data_length": len(x[0]),
            "out_data_length": len(y[0]),
        }
    else:
        dataset_config = {}

    return np.array(x), np.array(y), dataset_config


def process_folder(bvh_path, punch_labels_path, frame_rate_division, forward_direction, traj_window_root,
                   traj_window_wrist, num_tr_sampling_pts, develop):
    """
    Generates and stacks the X and Y data for provided files in the bvh and punch label folders.

    :@param bvh_path: str
    :@param punch_labels_path: str
    :@param frame_rate_div: int, # if 2, Reduces fps from 120fps to 60fps (60 fps reqd. for Axis Neuron bvh)
    :@param forward_direction: np.array, example: np.array([0.0, 0.0, 1.0]) for Z axis
    :@param window: int, number of frames in trajectory window
    :@param develop: boolean, flag for minimizing computations and enabling debug statements
    :@return:
    x_per_folder: np.array(n_frames_for_all_files, n_x_cols)
    y_per_folder: np.array(n_frames_for_all_files, n_y_cols)
    dataset_config: dict, containing parameters used to generate dataset and some info for accessing different
     variables in input and output vectors.
    """
    bvh_files = get_files_in_folder(bvh_path)
    punch_labels_files = get_files_in_folder(punch_labels_path)
    punch_stats = gen_punch_stats(punch_labels_path)

    x_per_folder = []
    y_per_folder = []

    if develop:
        bvh_files = [bvh_files[0]]
        punch_labels_files = [punch_labels_files[0]]

    for b_f, p_f in zip(bvh_files, punch_labels_files):
        print('\n')
        print(b_f, '\n', p_f)
        handler = FeatureExtractor(b_f, traj_window_root, traj_window_wrist, forward_dir=forward_direction,
                                   num_traj_sampling_pts=num_tr_sampling_pts)

        if b_f == bvh_files[-1]:
            x_cur_file, y_per_file, dataset_config = process_data(handler, p_f, frame_rate_division, forward_direction,
                                                                  traj_window_root, traj_window_wrist, develop=False,
                                                                  gen_data_config=True,
                                                                  punch_stats=punch_stats)

        else:
            x_cur_file, y_per_file, dataset_config = process_data(handler, p_f, frame_rate_division, forward_direction,
                                                                  traj_window_root, traj_window_wrist, develop=False)

        x_per_folder.append(x_cur_file)
        y_per_folder.append(y_per_file)
        if develop:
            break

    x_per_folder = np.vstack(x_per_folder)
    y_per_folder = np.vstack(y_per_folder)

    dataset_config["num_data_pts"] = x_per_folder.shape[0]

    return x_per_folder, y_per_folder, dataset_config
