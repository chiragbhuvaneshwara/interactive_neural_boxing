import math
import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
from data.neural_data_prep.nn_features.extractor import FeatureExtractor


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


def setup_output_dir(output_base_path, output_directory):
    """
    Sets up output dir if it doesn't exist. If it does exist, it empties the dir.
    :param output_base_path: str
    :param output_directory: str
    """
    if output_directory not in os.listdir(output_base_path):
        print('Creating new output dir:', output_directory)
        os.mkdir(os.path.join(output_base_path, output_directory))
    else:
        print('Emptying output dir:', output_directory)
        files = glob.glob(os.path.join(output_base_path, output_directory, '*'))
        for fi in files:
            os.remove(fi)


def process_data(handler: FeatureExtractor, punch_labels_csv_path, frame_rate_div):
    punch_targets_right = []
    punch_targets_left = []

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

        for i in range(handler.window_root, handler.n_frames - handler.window_root - 1, 1):
            if i % 50 == 0:
                print('Frames processed: ', i)
            punch_targets_right.append(handler.punch_targets[handler.hand_right][i].ravel())
            punch_targets_left.append(handler.punch_targets[handler.hand_left][i].ravel())

    return np.array(punch_targets_right), np.array(punch_targets_left)


def process_folder(bvh_path, punch_labels_path, frame_rate_division, forward_direction, traj_window_root,
                   traj_window_wrist, num_tr_sampling_pts):
    # bvh_files = get_files_in_folder(bvh_path)[:2]
    # punch_labels_files = get_files_in_folder(punch_labels_path)[:2]
    bvh_files = get_files_in_folder(bvh_path)
    punch_labels_files = get_files_in_folder(punch_labels_path)

    punch_targets_per_folder_right = []
    punch_targets_per_folder_left = []

    for b_f, p_f in zip(bvh_files, punch_labels_files):
        print('\n')
        print(b_f, '\n', p_f)
        handler = FeatureExtractor(b_f, traj_window_root, traj_window_wrist, forward_dir=forward_direction,
                                   num_traj_sampling_pts_root=num_tr_sampling_pts)

        punch_targets_cur_file_right, punch_targets_cur_file_left = process_data(handler, p_f, frame_rate_division)

        punch_targets_per_folder_right.append(punch_targets_cur_file_right)
        punch_targets_per_folder_left.append(punch_targets_cur_file_left)

    punch_targets_per_folder_right = np.vstack(punch_targets_per_folder_right)
    punch_targets_per_folder_left = np.vstack(punch_targets_per_folder_left)

    return punch_targets_per_folder_right, punch_targets_per_folder_left


INPUT_BASE_PATH = os.path.join("data", "raw_data")
OUTPUT_BASE_PATH = os.path.join("vis", "eval", "data")

PUNCH_LABELS_PATH = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "tertiary")
BVH_PATH = os.path.join(INPUT_BASE_PATH, "mocap", "hq", "processed")
FRAME_RATE_DIV = 1
FORWARD_DIR = np.array([0.0, 0.0, 1.0])
# TR_WINDOW = math.ceil(14 / FRAME_RATE_DIV)
TR_WINDOW_WRIST = math.ceil(5 / FRAME_RATE_DIV)
TR_WINDOW_ROOT = math.ceil(5 / FRAME_RATE_DIV)
TR_SAMPLES = 10
save_punch_targets_file = os.path.join(OUTPUT_BASE_PATH, "punch_targets_dataset_local_pos.npz")

####################### CONTROL PARAMS ###################################
try:
    punch_data_npz = np.load(save_punch_targets_file)
    punch_targets_dataset_right = punch_data_npz["punch_targets_dataset_right"]
    punch_targets_dataset_left = punch_data_npz["punch_targets_dataset_left"]
except:
    punch_targets_dataset_right, punch_targets_dataset_left = process_folder(BVH_PATH, PUNCH_LABELS_PATH,
                                                                             FRAME_RATE_DIV,
                                                                             FORWARD_DIR,
                                                                             TR_WINDOW_ROOT, TR_WINDOW_WRIST,
                                                                             TR_SAMPLES)

    punch_targets_dataset_right = punch_targets_dataset_right[
        np.argwhere(np.all(punch_targets_dataset_right != 0, axis=1)).ravel()]
    punch_targets_dataset_left = punch_targets_dataset_left[
        np.argwhere(np.all(punch_targets_dataset_left != 0, axis=1)).ravel()]

    np.savez(save_punch_targets_file, punch_targets_dataset_right=punch_targets_dataset_right,
             punch_targets_dataset_left=punch_targets_dataset_left)

punch_targets_dataset_right[:, [1, 2]] = punch_targets_dataset_right[:, [2, 1]]
punch_targets_dataset_left[:, [1, 2]] = punch_targets_dataset_left[:, [2, 1]]

punch_targets_dataset_right[:, [0, 1]] = punch_targets_dataset_right[:, [1, 0]]
punch_targets_dataset_left[:, [0, 1]] = punch_targets_dataset_left[:, [1, 0]]
fig = pyplot.figure()
ax = Axes3D(fig)

x, y, z = punch_targets_dataset_right[:, 0], punch_targets_dataset_right[:, 1], punch_targets_dataset_right[:, 2]
minimums = [i.min() for i in [x, y, z]]
maximums = [i.max() for i in [x, y, z]]

print("Right target details:")
print("Mins : ", minimums)
print("Maxs : ", maximums)

ax.scatter(x, y, z, c='r', marker='o')
# ax.set_xlabel('X axis (lateral direction)')
# ax.set_ylabel('Z axis (forward direction)')
# ax.set_zlabel('Y axis')
ax.set_ylabel('X axis (lateral direction)')
ax.set_xlabel('Z axis (forward direction)')
ax.set_zlabel('Y axis')
pyplot.savefig(os.path.join(OUTPUT_BASE_PATH, "punch_right_data.png"))

fig = pyplot.figure()
ax = Axes3D(fig)

x, y, z = punch_targets_dataset_left[:, 0], punch_targets_dataset_left[:, 1], punch_targets_dataset_left[:, 2]
minimums = [i.min() for i in [x, y, z]]
maximums = [i.max() for i in [x, y, z]]

print("Left target details:")
print("Mins : ", minimums)
print("Maxs : ", maximums)

ax.scatter(x, y, z, c='r', marker='x')
# ax.set_xlabel('X axis (lateral direction)')
# ax.set_ylabel('Z axis (forward direction)')
# ax.set_zlabel('Y axis')
ax.set_ylabel('X axis (lateral direction)')
ax.set_xlabel('Z axis (forward direction)')
ax.set_zlabel('Y axis')
pyplot.savefig(os.path.join(OUTPUT_BASE_PATH, "punch_left_data.png"))
