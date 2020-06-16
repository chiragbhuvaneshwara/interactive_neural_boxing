from mosi_utils_anim_t.preprocessing.NN_Features import FeatureExtractor, retrieve_name
import numpy as np
import pandas as pd


def prepare_indices_dict(**args):
    """
    :param args: all the variables that can be possibly used as part of X or Y
    :return:

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

    # print('keys: ',len(keys))
    print('col names len: ', len(col_names))
    # print(col_names)

    return indices_dict, col_names


def process_data(handler: FeatureExtractor, punch_p_csv_path):
    xc, yc = [], []

    frame_rate_div = 1

    for div in range(frame_rate_div):

        # Loading Bvh file into memory
        handler.load_motion(frame_rate_divisor=frame_rate_div, frame_rate_offset=div)

        # (n_frames, 2) => punch phase right, punch phase left
        punch_phase, punch_dphase = handler.load_punch_phase(punch_p_csv_path, frame_rate_divisor=frame_rate_div,
                                                             frame_rate_offset=div)

        right_punch_target = handler.get_punch_targets(punch_phase[:, 0], hand='right', phase_type='detailed')
        left_punch_target = handler.get_punch_targets(punch_phase[:, 1], hand='left', phase_type='detailed')

        #############################################################################################
        # These work but they aren't as accurate as blender

        local_positions = handler.get_root_local_joint_positions()
        local_velocities = handler.get_root_local_joint_velocities()

        root_velocity = handler.get_root_velocity()
        # root_rvelocity = handler.get_rotational_velocity()
        root_new_forward = handler.get_new_forward_dirs()

        feet_l, feet_r = handler.get_foot_concats()
        #############################################################################################

        indices_dict_set = False
        # for i in range(handler.window, handler.n_frames - handler.window, 1):
        for i in range(handler.window, 100 - handler.window, 1):

            if i % 50 == 0:
                print('Frames processed: ', i)

            traj_info = handler.get_trajectory(i)

            x_rootposs_tr = traj_info['rootposs']
            x_rootposs_tr = np.hstack(
                [x_rootposs_tr[:, 0].ravel(), x_rootposs_tr[:, 2].ravel()])  # Trajectory Pos, 2 * 12d

            x_rootvels_tr = traj_info['rootvels']
            x_rootvels_tr = np.hstack(
                [x_rootvels_tr[:, 0].ravel(), x_rootvels_tr[:, 2].ravel()])  # Trajectory Vels, 2 * 12d

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
                x_rootposs_tr,
                x_rootvels_tr,
                x_right_wrist_pos_tr,
                x_left_wrist_pos_tr,
                x_right_wrist_vels_tr,
                x_left_wrist_vels_tr,
                x_punch_phase,
                x_right_punch_target,
                x_left_punch_target,
                x_local_pos,
                x_local_vel
            ]

            keys = list(map(retrieve_name, x_curr_frame))
            kwargs = {k: v for k, v in zip(keys, x_curr_frame)}

            # rootposs, left_wrist_pos, right_wrist_pos, head_pos, rootdirs, headdirs, rootvels, left_wristvels, \
            # right_wristvels = handler.get_trajectory(i)
            # x_curr_frame = [
            #     rootposs[:, 0].ravel(), rootposs[:, 2].ravel(),  # Trajectory Pos, 2 * 12d
            #     # rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(),  # Trajectory Dir, 2 * 12d
            #     rootvels[:, 0].ravel(), rootvels[:, 2].ravel(),
            #     # rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait, 6 * 12d
            #     # rootgait[:,2].ravel(), rootgait[:,3].ravel(),
            #     # rootgait[:,4].ravel(), rootgait[:,5].ravel(),
            #     right_wrist_pos.ravel(),  # Trajectory Pos of wrist
            #     left_wrist_pos.ravel(),  # Trajectory Pos of wrist
            #     right_wristvels.ravel(),  # Trajectory Vels of wrist
            #     left_wristvels.ravel(),  # Trajectory Vels of wrist
            #     punch_phase[i].ravel(),  # Right punch phase followed by left punch phase
            #     right_punch_target[i].ravel(),
            #     left_punch_target[i].ravel(),
            #     local_positions[i - 1].ravel(),  # Joint Pos
            #     local_velocities[i - 1].ravel(),  # Joint Vel
            # ]

            if not indices_dict_set:
                x_indices, x_col_names = prepare_indices_dict(**kwargs)
                print('curr ', len(np.hstack(x_curr_frame)))
                print(x_indices)
                print('\n')

            xc.append(np.hstack(x_curr_frame))

            ############################################################################
            traj_info_next = handler.get_trajectory(i + 1, i + 1)

            y_rootposs_tr = traj_info_next['rootposs']
            y_rootposs_tr = np.hstack(
                [y_rootposs_tr[:, 0].ravel(), y_rootposs_tr[:, 2].ravel()])  # Trajectory Pos, 2 * 12d

            y_rootvels_tr = traj_info_next['rootvels']
            y_rootvels_tr = np.hstack(
                [y_rootvels_tr[:, 0].ravel(), y_rootvels_tr[:, 2].ravel()])  # Trajectory Vels, 2 * 12d

            y_right_wrist_pos_tr = traj_info_next['right_wrist_pos'].ravel()
            y_left_wrist_pos_tr = traj_info['left_wrist_pos'].ravel()

            y_right_wrist_vels_tr = traj_info_next['right_wristvels'].ravel()
            y_left_wrist_vels_tr = traj_info['left_wristvels'].ravel()

            y_punch_phase = punch_phase[i + 1].ravel()  # Right punch phase followed by left punch phase

            y_right_punch_target = right_punch_target[i + 1].ravel()
            y_left_punch_target = left_punch_target[i + 1].ravel()

            y_local_pos = local_positions[i].ravel()
            y_local_vel = local_velocities[i].ravel()

            y_root_velocity = np.hstack([root_velocity[i, 0, 0].ravel(), root_velocity[i, 0, 2].ravel()])
            y_root_new_forward = root_new_forward[i].ravel()
            y_punch_dphase = punch_dphase[i].ravel()
            y_foot_contacts = np.concatenate([feet_l[i], feet_r[i]], axis=-1)

            y_curr_frame = [
                y_root_velocity,
                y_root_new_forward,
                y_punch_dphase,
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

            # rootposs_next, left_wrist_pos_next, right_wrist_pos_next, head_pos_next, rootdirs_next, headdirs_next, rootvels_next, left_wristvels_next, right_wristvels_next = handler.get_trajectory(
            #     i + 1, i + 1)
            # y_curr_frame = [
            #     root_velocity[i, 0, 0].ravel(),  # Root Vel X, 1D
            #     root_velocity[i, 0, 2].ravel(),  # Root Vel Y, 1D
            #     # root_rvelocity[i].ravel(),    # Root Rot Vel, 1D
            #     root_new_forward[i].ravel(),  # new forward direction in 2D relative to past rotation.
            #     punch_dphase[i].ravel(),
            #     # dphase[i],                    # Change in Phase, 1D
            #     np.concatenate([feet_l[i], feet_r[i]], axis=-1),  # Contacts, 4D
            #     rootposs_next[:, 0].ravel(), rootposs_next[:, 2].ravel(),  # Next Trajectory Pos
            #     # rootdirs_next[:, 0].ravel(), rootdirs_next[:, 2].ravel(),  # Next Trajectory Dir
            #     rootvels_next[:, 0].ravel(), rootvels_next[:, 2].ravel(),
            #     right_wrist_pos_next.ravel(),  # Trajectory Pos of wrist
            #     left_wrist_pos_next.ravel(),  # Trajectory Pos of wrist
            #     right_wristvels_next.ravel(),  # Trajectory Vels of wrist
            #     left_wristvels_next.ravel(),  # Trajectory Vels of wrist
            #     local_positions[i].ravel(),  # Joint Pos
            #     local_velocities[i].ravel(),  # Joint Vel
            # ]

            if not indices_dict_set:
                y_indices, y_col_names = prepare_indices_dict(**kwargs)
                print('curr frame len ', len(np.hstack(y_curr_frame)))
                print(y_indices)
                print('\n')

            yc.append(np.hstack(y_curr_frame))

            indices_dict_set = True

    dataset_config = {
        "endJoints": 0,
        "numJoints": len(handler.joint_indices_dict.keys()),  # 59
        "use_rotations": False,
        "n_gaits": 1,
        "use_footcontacts": True,
        "foot_left": handler.foot_left,
        "foot_right": handler.foot_right,
        "zero_posture": handler.reference_skeleton,
        "joint_indices": handler.joint_indices_dict,
        "col_indices": [x_indices, y_indices],
        "col_names": [x_col_names, y_col_names]
    }

    return np.array(xc), np.array(yc), dataset_config


punch_phase_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/Punch.csv'
bvh_path = "C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Data/MocapBoxing/axis_neuron_processed/5_Punching_AxisNeuronProcessed_Char00.bvh"

handler = FeatureExtractor(bvh_path)
# manually set the skeleton parameters by manually checking the bvh files
handler.set_neuron_parameters()
handler.window = 25

####################################################################################
Xc, Yc, Dataset_Config = process_data(handler, punch_phase_path)

output_file_name = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-MOSI-DEV-VINN/mosi_dev_vinn/data/boxing'

X_train = Xc
Y_train = Yc
print('X_train shape: ', X_train.shape)
print('Y_train shape: ', Y_train.shape)
np.savez_compressed(output_file_name + "_train", Xun=X_train, Yun=Y_train)

X_train_df = pd.DataFrame(data=X_train, columns=Dataset_Config['col_names'][0])
X_train_df.to_csv(output_file_name + '_X.csv')
Y_train_df = pd.DataFrame(data=Y_train, columns=Dataset_Config['col_names'][1])
Y_train_df.to_csv(output_file_name + '_Y.csv')

print("done")
