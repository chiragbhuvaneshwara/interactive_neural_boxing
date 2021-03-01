import os
from nn_features import FeatureExtractor, retrieve_name
import numpy as np


# TODO Come up with better name below
def prepare_indices_dict(**args):
    """
    :param args: all the variables that can be possibly used as part of X or Y
    :return:
        indices_dict: dict of joint names with start_idx and end_idx
        col_names: list of col names with for example position marked as position_0, position_1, position_2 for x,y,z co-ordinates
    """
    keys = args.keys()
    col_names = []

    start = 0
    end = 0
    indices_dict = {}
    for k, v in zip(keys, args.values()):
        end = len(v) + end
        indices_dict[k] = [start, end]
        col_names_subset = [k] * (end - start)
        col_names_subset = [var + '_' + str(i) for i, var in enumerate(col_names_subset)]
        col_names.extend(col_names_subset)
        start = end

    print('col names len: ', len(col_names))

    return indices_dict, col_names


def prepare_input_data(i, handler, punch_phase, right_punch_target, left_punch_target, local_positions,
                       local_velocities, indices_dict_set=True):
    #TODO Add docstring after simplifying inputs i.e after moving all these input vars inside the handler
    traj_info = handler.get_trajectory(i)

    x_rootposs_tr = traj_info['rootposs']
    x_rootposs_tr = np.delete(x_rootposs_tr, 1, 1).ravel()  # Trajectory Pos, 2 * 10d

    x_rootvels_tr = traj_info['rootvels']
    x_rootvels_tr = np.delete(x_rootvels_tr, 1, 1).ravel()

    x_right_wrist_pos_tr = traj_info['right_wrist_pos'].ravel()
    x_left_wrist_pos_tr = traj_info['left_wrist_pos'].ravel()

    x_right_wrist_vels_tr = traj_info['right_wristvels'].ravel()
    x_left_wrist_vels_tr = traj_info['left_wristvels'].ravel()

    x_punch_phase = punch_phase[i].ravel()  # Right punch phase followed by left punch phase

    x_right_punch_target = right_punch_target[i].ravel()
    x_left_punch_target = left_punch_target[i].ravel()

    x_local_pos = local_positions[i - 1].ravel()
    x_local_vel = local_velocities[i - 1].ravel()

    x_curr_frame = [
        x_rootposs_tr,  # local wrt r in mid frame
        x_rootvels_tr,
        x_right_wrist_pos_tr,  # local wrt r in mid frame (TODO then wrt wrist in mid frame)
        x_left_wrist_pos_tr,  # local wrt r in mid frame (TODO then wrt wrist in mid frame)
        x_right_wrist_vels_tr,
        x_left_wrist_vels_tr,
        x_punch_phase,
        x_right_punch_target,  # local wrt r in mid frame
        x_left_punch_target,  # local wrt r in mid frame
        x_local_pos,
        x_local_vel
    ]

    if not indices_dict_set:
        keys = list(map(retrieve_name, x_curr_frame))
        kwargs = {k: v for k, v in zip(keys, x_curr_frame)}
        x_idxs, x_col_names = prepare_indices_dict(**kwargs)
        print('curr frame len ', len(np.hstack(x_curr_frame)))
        print(x_idxs)
        print('\n')
        return x_idxs, x_col_names
    else:
        return x_curr_frame


def prepare_output_data(i, handler, punch_phase, right_punch_target, left_punch_target, local_positions,
                        local_velocities, root_velocity, punch_dphase, feet_l, feet_r, root_new_forward,
                        indices_dict_set=True):
    #TODO Add docstring after simplifying inputs i.e after moving all these input vars inside the handler

    traj_info_next = handler.get_trajectory(i + 1, i + 1)

    y_rootposs_tr = traj_info_next['rootposs']
    y_rootposs_tr = np.delete(y_rootposs_tr, 1, 1).ravel()

    y_rootvels_tr = traj_info_next['rootvels']
    y_rootvels_tr = np.delete(y_rootvels_tr, 1, 1).ravel()

    y_right_wrist_pos_tr = traj_info_next['right_wrist_pos'].ravel()
    y_left_wrist_pos_tr = traj_info_next['left_wrist_pos'].ravel()

    y_right_wrist_vels_tr = traj_info_next['right_wristvels'].ravel()
    y_left_wrist_vels_tr = traj_info_next['left_wristvels'].ravel()

    y_punch_phase = punch_phase[i + 1].ravel()  # Right punch phase followed by left punch phase

    y_right_punch_target = right_punch_target[i + 1].ravel()
    y_left_punch_target = left_punch_target[i + 1].ravel()

    y_local_pos = local_positions[i].ravel()
    y_local_vel = local_velocities[i].ravel()

    y_root_velocity = np.hstack([root_velocity[i, 0, 0].ravel(), root_velocity[i, 0, 2].ravel()])
    y_root_new_forward = root_new_forward[i].ravel()

    # Taking i because length of phase is l and length of dphase is l-1
    y_punch_dphase = punch_dphase[i].ravel()

    y_foot_contacts = np.concatenate([feet_l[i], feet_r[i]], axis=-1)

    y_curr_frame = [
        y_root_velocity,
        y_root_new_forward,
        # y_punch_dphase,
        y_punch_phase,
        y_foot_contacts,
        y_rootposs_tr,
        y_rootvels_tr,
        y_right_wrist_pos_tr,
        y_left_wrist_pos_tr,
        y_right_wrist_vels_tr,
        y_left_wrist_vels_tr,
        y_local_pos,
        y_local_vel
    ]

    if not indices_dict_set:
        keys = list(map(retrieve_name, y_curr_frame))
        kwargs = {k: v for k, v in zip(keys, y_curr_frame)}
        y_idxs, y_col_names = prepare_indices_dict(**kwargs)
        print('curr frame len ', len(np.hstack(y_curr_frame)))
        print(y_idxs)
        print('\n')
        return y_idxs, y_col_names
    else:
        return y_curr_frame


def process_data(handler: FeatureExtractor, punch_p_csv_path, frame_rate_div, develop):
    """
    Generates and stacks the X and Y data for one provided bvh file and punch phase file.
    :param handler: FeatureExtractor,
    :param punch_p_csv_path: str,
    :param frame_rate_div: str,
    :param develop: boolean
    :return:
        x: np.array(n_frames_file, n_x_cols)
        y: np.array(n_frames_file, n_y_cols)
        dataset_config: dict
    """
    x, y = [], []

    for div in range(frame_rate_div):
        print('Processing Blender csv data %s' % frame_rate_div, div)
        handler.set_awinda_parameters()  # manually set the skeleton parameters by manually checking the bvh files
        # Loading Bvh file into memory
        handler.load_motion(frame_rate_divisor=frame_rate_div, frame_rate_offset=div)
        # (n_frames, 2) => punch phase right, punch phase left

        # TODO Save all handler generated vars before for loop inside the handler
        punch_phase, punch_dphase = handler.load_punch_phase(punch_p_csv_path, frame_rate_divisor=frame_rate_div,
                                                             frame_rate_offset=div)

        # TODO Only implemented for action label type tertiary currently. Must do binary labels and phase.
        right_punch_target = handler.get_punch_targets(punch_phase[:, 0], hand='right', space="local")
        left_punch_target = handler.get_punch_targets(punch_phase[:, 1], hand='left', space="local")

        local_positions = handler.get_root_local_joint_positions()
        local_velocities = handler.get_root_local_joint_velocities()

        root_velocity = handler.get_root_velocity()
        # root_rvelocity = handler.get_rotational_velocity()
        root_new_forward = handler.get_new_forward_dirs()

        feet_l, feet_r = handler.get_foot_concats()

        indices_dict_set = False
        x_indices, x_col_names = prepare_input_data(handler.window, handler, punch_phase, right_punch_target, left_punch_target,
                                                    local_positions, local_velocities, indices_dict_set)
        y_indices, y_col_names = prepare_output_data(handler.window, handler, punch_phase, right_punch_target, left_punch_target,
                                                     local_positions,
                                                     local_velocities, root_velocity, punch_dphase, feet_l, feet_r,
                                                     root_new_forward,
                                                     indices_dict_set)
        for i in range(handler.window, handler.n_frames - handler.window - 1, 1):
            if i % 50 == 0:
                print('Frames processed: ', i)
                if develop:
                    break
            x_curr_frame = prepare_input_data(i, handler, punch_phase, right_punch_target, left_punch_target,
                                              local_positions, local_velocities)
            x.append(np.hstack(x_curr_frame))

            y_curr_frame = prepare_output_data(i, handler, punch_phase, right_punch_target, left_punch_target,
                                               local_positions,
                                               local_velocities, root_velocity, punch_dphase, feet_l, feet_r,
                                               root_new_forward)
            y.append(np.hstack(y_curr_frame))

    dataset_config = {
        "endJoints": 0,
        "numJoints": len(handler.joint_indices_dict.keys()),
        "use_rotations": False,
        "n_gaits": 1,
        "use_footcontacts": True,
        "frd": frame_rate_div,
        "window": handler.window,
        "num_traj_samples": handler.num_traj_sampling_pts,
        "traj_step": handler.traj_step,
        # "foot_left": handler.foot_left,
        # "foot_right": handler.foot_right,
        "zero_posture": handler.reference_skeleton,
        "joint_indices": handler.joint_indices_dict,
        "col_indices": [x_indices, y_indices],
        "col_names": [x_col_names, y_col_names]
    }

    return np.array(x), np.array(y), dataset_config


def get_files_in_folder(folder):
    """
    :param folder: str
    :return:
        files: list, sorted
    """

    files = []
    dir_files = os.listdir(folder)
    dir_files.sort()
    for path in dir_files:
        full_path = os.path.join(folder, path)
        files.append(full_path)

    return files


def process_folder(bvh_path, punch_phase_path, frame_rate_div, forward_direction, window, develop):
    """
    Generates and stacks the X and Y data for provided files in the bvh and punch phase folders.
    :param bvh_path: str
    :param punch_phase_path: str
    :param frame_rate_div: int, # if 2, Reduces fps from 120fps to 60fps (60 fps reqd. for Axis Neuron bvh)
    :param forward_direction: np.array, example: np.array([0.0, 0.0, 1.0]) for Z axis
    :param window: int, number of frames in trajectory window
    :param develop: boolean, flag for minimizing computations and enabling debug statements
    :return:
        x_per_folder: np.array(n_frames_for_all_files, n_x_cols)
        y_per_folder: np.array(n_frames_for_all_files, n_y_cols)
        dataset_config: dict
    """
    bvh_files = get_files_in_folder(bvh_path)
    punch_phase_files = get_files_in_folder(punch_phase_path)

    x_per_folder = []
    y_per_folder = []
    for b_f, p_f in zip(bvh_files, punch_phase_files):
        print(b_f, '\n', p_f)
        handler = FeatureExtractor(b_f, window, forward_dir=forward_direction)
        x_cur_file, y_per_file, dataset_config = process_data(handler, p_f, frame_rate_div, develop)
        x_per_folder.append(x_cur_file)
        y_per_folder.append(y_per_file)
        if develop:
            break

    x_per_folder = np.vstack(x_per_folder)
    y_per_folder = np.vstack(y_per_folder)

    return x_per_folder, y_per_folder, dataset_config