from mosi_utils_anim.preprocessing.NN_Features import FeatureExtractor
import numpy as np
import glob, os, json
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing


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


def process_data(handler: FeatureExtractor, blender_path):
    bvh_path = handler.bvh_file_path
    phase_path = bvh_path.replace('.bvh', '.phase')
    gait_path = bvh_path.replace(".bvh",
                                 ".gait")  # upload the punch detection csv or numpy here, how to interpret must be
    # decided by you blender_path = 'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Blender Code
    # Snippets\data annotation res\Punch.csv'
    Pc, Xc, Yc = [], [], []

    frame_rate_div = 1

    for div in range(frame_rate_div):
        handler.load_motion(frame_rate_divisor=frame_rate_div, frame_rate_offset=div)
        # gait = handler.load_gait(gait_path, frame_rate_divisor = frame_rate_div, frame_rate_offset=div)
        # phase, dphase = handler.load_phase(phase_path, frame_rate_divisor = frame_rate_div, frame_rate_offset=div)
        blender_np = handler.load_blender_data(blender_path, frame_rate_divisor=frame_rate_div, frame_rate_offset=div)
        # right_punch_target, right_action = handler.get_punch_targets(blender_np[:, 0])
        right_punch_target = handler.get_punch_targets(blender_np[:, 0])
        # left_punch_target, left_action = handler.get_punch_targets(blender_np[:, 1])
        left_punch_target = handler.get_punch_targets(blender_np[:, 1])
        #############################################################################################
        # These work but they aren't as accurate as blender

        local_positions = handler.get_root_local_joint_positions()
        local_velocities = handler.get_root_local_joint_velocities()

        root_velocity = handler.get_root_velocity()
        # root_rvelocity = handler.get_rotational_velocity()
        root_new_forward = handler.get_new_forward_dirs()

        feet_l, feet_r = handler.get_foot_concats()
        #############################################################################################

        for i in range(handler.window, handler.n_frames - handler.window, 1):
            rootposs, rootdirs = handler.get_trajectory(i)
            # rootgait = gait[i - handler.window:i+handler.window:10]
            #
            # Pc.append(phase[i])

            # rows_to_keep = [el for el in range(i - handler.window, i + handler.window, 10)]
            # print(rows_to_keep)
            # blender_data = np.take(blender_np, rows_to_keep, axis=0)  # 6 (rows) * 2d (left punch and right punch)
            blender_data = blender_np[i]
            # print('#################')
            # print(i)
            # print('#################')

            Xc.append(np.hstack([
                # rootposs[:, 0].ravel(), rootposs[:, 2].ravel(),  # Trajectory Pos, 2 * 12d
                # rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(),  # Trajectory Dir, 2 * 12d
                # rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait, 6 * 12d
                # rootgait[:,2].ravel(), rootgait[:,3].ravel(),
                # rootgait[:,4].ravel(), rootgait[:,5].ravel(),
                blender_data.ravel(),
                # 2d (left punch phase and right punch phase)  (Only giving how far we are into the punch action. What if we have more actions? _Need an action vector)
                # right_action[i],
                # left_action[i],
                right_punch_target[i].ravel(),
                left_punch_target[i].ravel(),
                local_positions[i - 1].ravel(),  # Joint Pos
                local_velocities[i - 1].ravel(),  # Joint Vel
            ]))

            rootposs_next, rootdirs_next = handler.get_trajectory(i + 1, i + 1)

            Yc.append(np.hstack([
                root_velocity[i, 0, 0].ravel(),  # Root Vel X, 1D
                root_velocity[i, 0, 2].ravel(),  # Root Vel Y, 1D
                # root_rvelocity[i].ravel(),    # Root Rot Vel, 1D
                root_new_forward[i].ravel(),  # new forward direction in 2D relative to past rotation.
                # dphase[i],                    # Change in Phase, 1D
                # right_punch_target.ravel(),
                # left_punch_target.ravel(),
                np.concatenate([feet_l[i], feet_r[i]], axis=-1),  # Contacts, 4D
                # rootposs_next[:, 0].ravel(), rootposs_next[:, 2].ravel(),  # Next Trajectory Pos
                # rootdirs_next[:, 0].ravel(), rootdirs_next[:, 2].ravel(),  # Next Trajectory Dir
                local_positions[i].ravel(),  # Joint Pos
                local_velocities[i].ravel(),  # Joint Vel
            ]))

    dataset_config = {
        "endJoints": 0,
        "numJoints": 31,
        "use_rotations": False,
        "n_gaits": 6,
        "use_footcontacts": True,
        "foot_left": handler.foot_left,
        "foot_right": handler.foot_right,
        "zero_posture": handler.reference_skeleton
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

blender_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/Punch.csv'
bvh_path = "C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Data/MocapBoxing/axis_neuron_processed/5_Punching_AxisNeuronProcessed_Char00.bvh"
data_folder = "./test_files"
patches_path = "./test_files/patches.npz"

# Pc_old, Xc_old, Yc_old = old_preprocessing.generate_database(data_folder)


handler = FeatureExtractor(bvh_path)
# handler.set_holden_parameters()
# handler.set_makehuman_parameters()  #manually set the skeleton parameters by manually checking the bvh files
handler.set_neuron_parameters()
handler.window = 60
# handler.n_joints = 31
# handler.load_motion()

####################################################################################
# TO DECIDE: phase required or not
# Pc, Xc, Yc, dataset_config = process_data(handler, blender_path)
Pc, Xc, Yc, dataset_config = process_data(handler, blender_path)

# pd.DataFrame(Xc).to_csv(
#     "C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/nn_features_input.csv")
# pd.DataFrame(Yc).to_csv(
#     "C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/nn_features_output.csv")

output_file_name = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-MOSI-DEV-VINN/mosi_dev_vinn/data/boxing'

# Xun = np.concatenate(Xtrain, axis=0)
# Yun = np.concatenate(Ytrain, axis=0)
# Pun = np.concatenate(Ptrain, axis=0)

# print(Xun.shape, Yun.shape, Pun.shape)
X_train = Xc[:4000,:]
Y_train = Yc[:4000,:]
print(X_train.shape)
print(Y_train.shape)
np.savez_compressed(output_file_name + "_train", Xun=X_train, Yun=Y_train)

X_test = Xc[4000:,:]
Y_test = Yc[4000:,:]
print(X_test.shape)
print(Y_test.shape)
np.savez_compressed(output_file_name + "_test", Xun=X_test, Yun=Y_test)

with open(output_file_name + "_config.json", "w") as f:
    json.dump(dataset_config, f)



# xslice = slice(((handler.window*2)//10)*10+1, ((handler.window*2)//10)*10+handler.n_joints*3+1, 3)
# yslice = slice(8+(handler.window//10)*4+1, 8+(handler.window//10)*4+handler.n_joints*3+1, 3)
# X, Y, P, config = PREPROCES S_FOLDER(data_folder, "data_4D_60fps", handler, process_data, False, xslice, yslice, patches_path)

print("done")
