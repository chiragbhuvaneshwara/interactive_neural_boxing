import glob
import os
import math
from mosi_utils_anim_t.preprocessing.NN_Features import FeatureExtractor, retrieve_name
import numpy as np
import pandas as pd
import json


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
        # indices_dict.append((k, [start, end]))
        indices_dict[k] = [start, end]
        col_names_subset = [k] * (end - start)
        col_names_subset = [var + '_' + str(i) for i, var in enumerate(col_names_subset)]
        col_names.extend(col_names_subset)
        start = end

    print('col names len: ', len(col_names))

    return indices_dict, col_names


def process_data(handler: FeatureExtractor, punch_p_csv_path, frame_rate_div):
    xc, yc = [], []

    for div in range(frame_rate_div):
        print('Processing Blender csv data %s' % frame_rate_div, div)
        # Loading Bvh file into memory
        handler.load_motion(frame_rate_divisor=frame_rate_div, frame_rate_offset=div)
        # manually set the skeleton parameters by manually checking the bvh files
        handler.set_awinda_parameters()
        # (n_frames, 2) => punch phase right, punch phase left
        punch_phase, punch_dphase = handler.load_punch_phase(punch_p_csv_path, frame_rate_divisor=frame_rate_div,
                                                             frame_rate_offset=div)

        # TODO Only implemented for phase type tertiary currently
        right_punch_target = handler.get_punch_targets(punch_phase[:, 0], hand='right', phase_type='tertiary')
        left_punch_target = handler.get_punch_targets(punch_phase[:, 1], hand='left', phase_type='tertiary')

        #############################################################################################
        # These work but they aren't as accurate as blender
        local_positions = handler.get_root_local_joint_positions()
        local_velocities = handler.get_root_local_joint_velocities()

        root_velocity = handler.get_root_velocity()
        # root_rvelocity = handler.get_rotational_velocity()
        root_new_forward = handler.get_new_forward_dirs()

        feet_l, feet_r = handler.get_foot_concats()
        #############################################################################################
        indices_dict_set = False  # just to print out useful info
        for i in range(handler.window, handler.n_frames - handler.window - 1, 1):
        # for i in range(handler.window, 100 - handler.window - 1, 1):
            if i % 50 == 0:
                print('Frames processed: ', i)

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
                x_rootposs_tr,              # local wrt r in mid frame
                x_rootvels_tr,
                x_right_wrist_pos_tr,       # local wrt r in mid frame then wrt wrist in mid frame
                x_left_wrist_pos_tr,        # local wrt r in mid frame then wrt wrist in mid frame
                x_right_wrist_vels_tr,
                x_left_wrist_vels_tr,
                x_punch_phase,
                x_right_punch_target,       # local wrt r in mid frame
                x_left_punch_target,        # local wrt r in mid frame
                x_local_pos,
                x_local_vel
            ]

            keys = list(map(retrieve_name, x_curr_frame))
            kwargs = {k: v for k, v in zip(keys, x_curr_frame)}

            if not indices_dict_set:
                x_indices, x_col_names = prepare_indices_dict(**kwargs)
                print('curr frame len ', len(np.hstack(x_curr_frame)))
                print(x_indices)
                print('\n')

            xc.append(np.hstack(x_curr_frame))

            ############################################################################

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

            keys = list(map(retrieve_name, y_curr_frame))
            kwargs = {k: v for k, v in zip(keys, y_curr_frame)}

            if not indices_dict_set:
                y_indices, y_col_names = prepare_indices_dict(**kwargs)
                print('curr frame len ', len(np.hstack(y_curr_frame)))
                print(y_indices)
                print('\n')

            yc.append(np.hstack(y_curr_frame))

            indices_dict_set = True

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
        # "zero_posture": handler.reference_skeleton,
        "joint_indices": handler.joint_indices_dict,
        "col_indices": [x_indices, y_indices],
        "col_names": [x_col_names, y_col_names]
    }

    return np.array(xc), np.array(yc), dataset_config


input_base_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor'
punch_phase_path = input_base_path + '/Blender Code Snippets/data annotation res/new_data/tertiary/boxing_2_tertiary.csv'
bvh_path = input_base_path + "/Data/boxing_chirag/processed/boxing_2.bvh"
####################################################################################
frame_rate_div = 1  # if 2, Reduces fps from 120fps to 60fps for Axis Neuron bvh
# Ensure: Rotated the bvh mocap data to right hand co-ordinate system i.e. y up and z forward
forward_direction = np.array([0.0, 0.0, 1.0])  # Z axis
window = math.ceil(25 / frame_rate_div)
handler = FeatureExtractor(bvh_path, window, forward_dir=forward_direction)
Xc, Yc, Dataset_Config = process_data(handler, punch_phase_path, frame_rate_div)

output_base_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-MOSI-DEV-VINN/mosi_dev_vinn/data/'
frd_win = 'boxing_fr_' + str(frame_rate_div) + '_' + str(window)
if frd_win not in os.listdir(output_base_path):
    print('Creating new output dir:', frd_win)
    os.mkdir(os.path.join(output_base_path, frd_win))
    out_dir = os.path.join(output_base_path, frd_win)
else:
    print('Emptying output dir:', frd_win)
    files = glob.glob(os.path.join(output_base_path, frd_win, '*'))
    for f in files:
        os.remove(f)
    out_dir = os.path.join(output_base_path, frd_win)

X_train = Xc
Y_train = Yc
print('X_train shape: ', X_train.shape)
print('X_train mean: ', X_train.mean())
print('Y_train shape: ', Y_train.shape)
print('Y_train mean: ', Y_train.mean())
np.savez_compressed(os.path.join(out_dir, "train"), Xun=X_train, Yun=Y_train)

X_train_df = pd.DataFrame(data=X_train, columns=Dataset_Config['col_names'][0])
X_train_df.to_csv(os.path.join(out_dir, "X.csv"))
Y_train_df = pd.DataFrame(data=Y_train, columns=Dataset_Config['col_names'][1])
Y_train_df.to_csv(os.path.join(out_dir, "Y.csv"))

with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(Dataset_Config, f)

print("done")
