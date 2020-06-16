from mosi_utils_anim_t.preprocessing.NN_Features import FeatureExtractor, retrieve_name
import numpy as np
import glob, os, json
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing
import collections


def prepare_indices_dict(**args):

    keys = args.keys()
    col_names = []

    start = 0
    end = 0
    indices_dict = []
    for k, v in zip(keys, args.values()):
        end = len(v) + end
        indices_dict.append((k, [start, end]))
        col_names_subset = [k]*(end-start)
        col_names_subset = [var + '_' + str(i) for i, var in enumerate(col_names_subset)]
        col_names.extend(col_names_subset)
        start = end

    # print('keys: ',len(keys))
    print('names: ',len(col_names))
    # print(col_names)

    return indices_dict, col_names


def PREPROCESS_FOLDER(bvh_folder_path, output_file_name, base_handler: FeatureExtractor, process_data_function,
                      terrain_fitting=True, terrain_xslice=[], terrain_yslice=[], patches_path="", split_files=False):
    """
    This function processes a whole folder of BVH files. The "process_data_function" is called for each file and creates the actual training data (x,y,p). 
    process_data_function(handler : Preprocessing_handler) -> [P, X, Y]
    The handler starts as a copy from base_handler

    Pseudo-Code: 
        for bvh_file in folder:
            handler = base_handler.copy()
            handler.bvh_file = bvh_file
            handler.load_data()

            Pc, Xc, Yc = process_data_function(handler)
            merge (P,X,Y) (Pc, Xc, Yc)
        
        save file
        return X, Y, P

        :param bvh_folder_path: path to folder containing bvh files and labels
        :param output_file_name: output file to which the processed data should be written
        :param base_handler: base-handler containing configuration information (e.g. window size, # joints, etc.)
        :param process_data_function: process data function to create a single x-y pair (+p)
        :param terrain_fitting=True: If true, terrain fitting is applied
        :param terrain_xslice=[]: xslice definining joint positions in X that are affected by terrain fitting
        :param terrain_yslice=[]: yslice definining joint positions in Y that are affected by terrain fitting
        :param patches_path="": path to compressed patches file. 
        :param split_files=False: not yet implemented


    Example for a single file: 

        def process_data(handler):
            bvh_path = handler.bvh_file_path
            phase_path = bvh_path.replace('.bvh', '.phase')
            gait_path = bvh_path.replace(".bvh", ".gait")

            Pc, Xc, Yc = [], [], []


            gait = handler.load_gait(gait_path, adjust_crouch=True)
            phase, dphase = handler.load_phase(phase_path)

            local_positions = handler.get_root_local_joint_positions()
            local_velocities = handler.get_root_local_joint_velocities()

            root_velocity = handler.get_root_velocity()
            root_rvelocity = handler.get_rotational_velocity()

            feet_l, feet_r = handler.get_foot_concats()

            for i in range(handler.window, handler.n_frames - handler.window - 1, 1):
                rootposs,rootdirs = handler.get_trajectory(i)
                rootgait = gait[i - handler.window:i+handler.window:10]

                Pc.append(phase[i])

                Xc.append(np.hstack([
                        rootposs[:,0].ravel(), rootposs[:,2].ravel(), # Trajectory Pos
                        rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # Trajectory Dir
                        rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait
                        rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                        rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                        local_positions[i-1].ravel(),  # Joint Pos
                        local_velocities[i-1].ravel(), # Joint Vel
                    ]))

                rootposs_next, rootdirs_next = handler.get_trajectory(i + 1, i + 1)

                Yc.append(np.hstack([
                    root_velocity[i,0,0].ravel(), # Root Vel X
                    root_velocity[i,0,2].ravel(), # Root Vel Y
                    root_rvelocity[i].ravel(),    # Root Rot Vel
                    dphase[i],                    # Change in Phase
                    np.concatenate([feet_l[i], feet_r[i]], axis=-1), # Contacts
                    rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # Next Trajectory Pos
                    rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # Next Trajectory Dir
                    local_positions[i].ravel(),  # Joint Pos
                    local_velocities[i].ravel(), # Joint Vel
                    ]))

            return np.array(Pc), np.array(Xc), np.array(Yc)

        bvh_path = "./test_files/LocomotionFlat04_000.bvh"
        data_folder = "./test_files"
        patches_path = "./test_files/patches.npz"
        
        handler = FeatureExtractor
    (bvh_path)
        handler.load_motion()
        Pc, Xc, Yc = process_data(handler)
        
        xslice = slice(((handler.window*2)//10)*10+1, ((handler.window*2)//10)*10+handler.n_joints*3+1, 3)
        yslice = slice(8+(handler.window//10)*4+1, 8+(handler.window//10)*4+handler.n_joints*3+1, 3)
        P, X, Y = handler.terrain_fitting(bvh_path.replace(".bvh", '_footsteps.txt'), patches_path, Pc, Xc, Yc, xslice, yslice)

    """

    P, X, Y = [], [], []
    bvhfiles = glob.glob(os.path.join(bvh_folder_path, '*.bvh'))
    data_folder = bvh_folder_path
    print(bvhfiles, os.path.join(bvh_folder_path, '*.bvh'))
    config = {}

    def process_single(data):
        filename = os.path.split(data)[-1]
        data = os.path.join(data_folder, filename)

        handler = base_handler.copy()
        handler.bvh_file_path = data
        # handler.load_motion()

        Pc, Xc, Yc, config = process_data_function(handler)
        if terrain_fitting:
            Ptmp, Xtmp, Ytmp = Pc, Xc, Yc
            print("here would have been some terrain fitting")
            # Ptmp, Xtmp, Ytmp = handler.terrain_fitting(data.replace(".bvh", '_footsteps.txt'), patches_path, Pc, Xc, Yc, terrain_xslice, terrain_yslice)
        else:
            Ptmp, Xtmp, Ytmp = Pc, Xc, Yc
        return Xtmp, Ytmp, Ptmp, config, filename

    # for data in bvhfiles:
    num_cores = multiprocessing.cpu_count()
    tmp = Parallel(n_jobs=num_cores - 1)(delayed(process_single)(data) for data in bvhfiles)

    Ptmp, Xtmp, Ytmp = {}, {}, {}
    for r in tmp:
        Xtmp[r[4]] = r[0]
        Ytmp[r[4]] = r[1]
        Ptmp[r[4]] = r[2]
        config = r[3]

    Xtrain, Ytrain, Ptrain = [], [], []
    Xtest, Ytest, Ptest = [], [], []

    for r in tmp:
        if not "mirror" in r[4]:
            name = r[4]
            random_indizes = np.arange(len(r[0]))
            # np.random.shuffle(random_indizes)
            train_indizes = np.array(random_indizes[0:int(len(random_indizes) * 0.9)])
            test_indizes = np.array(random_indizes[int(len(random_indizes) * 0.9):])

            Xtrain.extend([Xtmp[name][train_indizes]])
            Ytrain.extend([Ytmp[name][train_indizes]])
            Ptrain.extend([Ptmp[name][train_indizes]])

            Xtest.extend([Xtmp[name][test_indizes]])
            Ytest.extend([Ytmp[name][test_indizes]])
            Ptest.extend([Ptmp[name][test_indizes]])

            # name = name.replace(".bvh", "_mirror.bvh")
            # Xtrain.extend([Xtmp[name][train_indizes]])
            # Ytrain.extend([Ytmp[name][train_indizes]])
            # Ptrain.extend([Ptmp[name][train_indizes]])

            # Xtest.extend([Xtmp[name][test_indizes]])
            # Ytest.extend([Ytmp[name][test_indizes]])
            # Ptest.extend([Ptmp[name][test_indizes]])

    # for r in tmp:
    #     Xtmp = r[0]
    #     Ytmp = r[1]
    #     Ptmp = r[2]
    #     config = r[3]
    #     P.extend([Ptmp])
    #     X.extend([Xtmp])
    #     Y.extend([Ytmp])

    """ Clip Statistics """
    print("Training")
    print('     Total Clips: %i' % len(Xtrain))
    print('     Shortest Clip: %i' % min(map(len, Xtrain)))
    print('     Longest Clip: %i' % max(map(len, Xtrain)))
    print('     Average Clip: %i' % np.mean(list(map(len, Xtrain))))

    print("Testing")
    print('     Total Clips: %i' % len(Xtest))
    print('     Shortest Clip: %i' % min(map(len, Xtest)))
    print('     Longest Clip: %i' % max(map(len, Xtest)))
    print('     Average Clip: %i' % np.mean(list(map(len, Xtest))))

    """ Merge Clips """

    print('Merging Clips...')

    Xun = np.concatenate(Xtrain, axis=0)
    Yun = np.concatenate(Ytrain, axis=0)
    Pun = np.concatenate(Ptrain, axis=0)

    print(Xun.shape, Yun.shape, Pun.shape)
    np.savez_compressed(output_file_name + "_train", Xun=Xun, Yun=Yun, Pun=Pun)

    Xun = np.concatenate(Xtest, axis=0)
    Yun = np.concatenate(Ytest, axis=0)
    Pun = np.concatenate(Ptest, axis=0)

    print(Xun.shape, Yun.shape, Pun.shape)
    np.savez_compressed(output_file_name + "_test", Xun=Xun, Yun=Yun, Pun=Pun)

    Xun = np.concatenate([np.concatenate(Xtrain, axis=0), np.concatenate(Xtest, axis=0)], axis=0)
    Yun = np.concatenate([np.concatenate(Ytrain, axis=0), np.concatenate(Ytest, axis=0)], axis=0)
    Pun = np.concatenate([np.concatenate(Ptrain, axis=0), np.concatenate(Ptest, axis=0)], axis=0)

    np.savez_compressed(output_file_name, Xun=Xun, Yun=Yun, Pun=Pun)

    with open(output_file_name + ".json", "w") as f:
        json.dump(config, f)
    return Xun, Yun, Pun, config


def process_data(handler: FeatureExtractor, punch_p_csv_path):
    bvh_path = handler.bvh_file_path
    # decided by you punch_p_csv_path = 'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Blender Code
    # Snippets\data annotation res\Punch.csv'
    Pc, Xc, Yc = [], [], []

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

            # rootposs, left_wrist_pos, right_wrist_pos, head_pos, rootdirs, headdirs, rootvels, left_wristvels, \
            # right_wristvels = handler.get_trajectory(i)

            traj_info = handler.get_trajectory(i)

            # rootgait = gait[i - handler.window:i+handler.window:10]
            # Pc.append(phase[i])

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

            X_curr_frame = [
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

            keys = list(map(retrieve_name, X_curr_frame))
            kwargs = {k: v for k, v in zip(keys, X_curr_frame)}

            # X_curr_frame = [
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
                # X_indices = prepare_indices_dict(*X_curr_frame)
                X_indices, X_col_names = prepare_indices_dict(**kwargs)
                print('curr ',len(np.hstack(X_curr_frame)))
                print(X_indices)

            Xc.append(np.hstack(X_curr_frame))

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

            Y_curr_frame = [
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

            keys = list(map(retrieve_name, Y_curr_frame))
            kwargs = {k: v for k, v in zip(keys, Y_curr_frame)}

            # rootposs_next, left_wrist_pos_next, right_wrist_pos_next, head_pos_next, rootdirs_next, headdirs_next, rootvels_next, left_wristvels_next, right_wristvels_next = handler.get_trajectory(
            #     i + 1, i + 1)
            # Y_curr_frame = [
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
                # X_indices = prepare_indices_dict(*X_curr_frame)
                Y_indices, Y_col_names = prepare_indices_dict(**kwargs)
                print('curr ',len(np.hstack(Y_curr_frame)))
                print(Y_indices)
            Yc.append(np.hstack(Y_curr_frame))

            indices_dict_set = True

    dataset_config = {
        "endJoints": 0,
        # "numJoints": 59,
        "numJoints": len(handler.joint_indices_dict.keys()), # 59
        "use_rotations": False,
        "n_gaits": 1,
        "use_footcontacts": True,
        "foot_left": handler.foot_left,
        "foot_right": handler.foot_right,
        "zero_posture": handler.reference_skeleton,
        "col_indices": [X_indices, Y_indices],
        "col_names": [X_col_names, Y_col_names]
    }

    return np.array(Pc), np.array(Xc), np.array(Yc), dataset_config
    # return np.array(Xc), np.array(Yc), dataset_config


from mosi_utils_anim.animation_data.utils import quaternion_from_matrix, euler_from_matrix, euler_matrix, \
    quaternion_matrix
import math


def print_degrees(a):
    b = np.zeros(3)
    for i in range(3):
        b[i] = math.degrees(a[i])
    return b


def rotation_to_target(vecA, vecB):
    vecA = vecA / np.linalg.norm(vecA)
    vecB = vecB / np.linalg.norm(vecB)
    dt = np.dot(vecA, vecB)
    cross = np.linalg.norm(np.cross(vecA, vecB))
    G = np.array([[dt, -cross, 0], [cross, dt, 0], [0, 0, 1]])

    v = (vecB - dt * vecA)
    v = v / np.linalg.norm(v)
    w = np.cross(vecB, vecA)
    # F = np.array([[vecA[0], vecA[1], vecA[2]], [v[0], v[1], v[2]], [w[0], w[1], w[2]]])
    F = np.array([vecA, v, w])

    # U = np.matmul(np.linalg.inv(F), np.matmul(G, F))
    U = np.matmul(np.matmul(np.linalg.inv(F), G), F)
    # U = np.zeros((4,4))
    # U[3,3] = 1
    # U[:3,:3] = b

    test = np.matmul(U, vecA)
    if np.linalg.norm(test - vecB) > 0.0001:
        print("error: ", test, vecB)

    # b = np.matmul(np.linalg.inv(F), np.matmul(G, F))
    b = np.matmul(np.matmul(np.linalg.inv(F), G), F)
    U = np.zeros((4, 4))
    U[3, 3] = 1
    U[:3, :3] = b
    q = quaternion_from_matrix(U)
    q[3] = -q[3]
    return q


# print(euler_matrix(math.radians(20.427), math.radians(56.301), math.radians(142.783)))

punch_phase_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/Punch.csv'
bvh_path = "C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Data/MocapBoxing/axis_neuron_processed/5_Punching_AxisNeuronProcessed_Char00.bvh"
data_folder = "./test_files"
patches_path = "./test_files/patches.npz"

handler = FeatureExtractor(bvh_path)
#manually set the skeleton parameters by manually checking the bvh files
handler.set_neuron_parameters()
handler.window = 25

####################################################################################
Pc, Xc, Yc, dataset_config = process_data(handler, punch_phase_path)

output_file_name = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-MOSI-DEV-VINN/mosi_dev_vinn/data/boxing'

X_train = Xc
Y_train = Yc
print('X_train shape: ', X_train.shape)
print('Y_train shape: ', Y_train.shape)
np.savez_compressed(output_file_name + "_train", Xun=X_train, Yun=Y_train)

X_train_df = pd.DataFrame(data=X_train, columns=dataset_config['col_names'][0])
X_train_df.to_csv(output_file_name+'_X.csv')
Y_train_df = pd.DataFrame(data=Y_train, columns=dataset_config['col_names'][1])
Y_train_df.to_csv(output_file_name+'_Y.csv')

# X_test = Xc[4000:,:]
# Y_test = Yc[4000:,:]
# print(X_test.shape)
# print(Y_test.shape)
# np.savez_compressed(output_file_name + "_test", Xun=X_test, Yun=Y_test)
# with open(output_file_name + "_config3.json", "w") as f:
#     json.dump(dataset_config, f)

print("done")

# xslice = slice(((handler.window*2)//10)*10+1, ((handler.window*2)//10)*10+handler.n_joints*3+1, 3)
# yslice = slice(8+(handler.window//10)*4+1, 8+(handler.window//10)*4+handler.n_joints*3+1, 3)
# X, Y, P, config = PREPROCES S_FOLDER(data_folder, "data_4D_60fps", handler, process_data, False, xslice, yslice, patches_path)


