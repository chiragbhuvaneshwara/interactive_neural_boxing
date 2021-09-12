import json
import math
import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from data.neural_data_prep.nn_features.extractor import FeatureExtractor
import eval.gen_punch_targets.utils.common as utils


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

        # calculate_punch_targets Only implemented for action label type tertiary currently
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
OUTPUT_BASE_PATH = os.path.join("eval", "saved")

PUNCH_LABELS_PATH = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "tertiary")
BVH_PATH = os.path.join(INPUT_BASE_PATH, "mocap", "hq", "processed")
FRAME_RATE_DIV = 1
FORWARD_DIR = np.array([0.0, 0.0, 1.0])
# TR_WINDOW = math.ceil(14 / FRAME_RATE_DIV)
TR_WINDOW_WRIST = math.ceil(5 / FRAME_RATE_DIV)
TR_WINDOW_ROOT = math.ceil(5 / FRAME_RATE_DIV)
TR_SAMPLES = 10
save_punch_targets_json = os.path.join(OUTPUT_BASE_PATH, "targets", "data",
                                       "dataset_punch_targets_local_pos_py_space.json")
save_test_path = os.path.join(OUTPUT_BASE_PATH, "targets", "test")

save_punch_targets_plot_path = os.path.join(OUTPUT_BASE_PATH, "plots")

####################### CONTROL PARAMS ###################################
try:
    with open(save_punch_targets_json) as json_file:
        punch_data = json.load(json_file)

    punch_targets_dataset_right = np.array(punch_data["punch_targets_dataset_right"])
    punch_targets_dataset_left = np.array(punch_data["punch_targets_dataset_left"])
except FileNotFoundError:
    punch_targets_dataset_right, punch_targets_dataset_left = process_folder(BVH_PATH, PUNCH_LABELS_PATH,
                                                                             FRAME_RATE_DIV,
                                                                             FORWARD_DIR,
                                                                             TR_WINDOW_ROOT, TR_WINDOW_WRIST,
                                                                             TR_SAMPLES)

    punch_targets_dataset_right = punch_targets_dataset_right[
        np.argwhere(np.all(punch_targets_dataset_right != 0, axis=1)).ravel()]
    punch_targets_dataset_left = punch_targets_dataset_left[
        np.argwhere(np.all(punch_targets_dataset_left != 0, axis=1)).ravel()]

    punch_data = {
        "punch_targets_dataset_right": punch_targets_dataset_right.tolist(),
        "punch_targets_dataset_left": punch_targets_dataset_left.tolist()
    }

    with open(save_punch_targets_json, 'w') as f:
        json.dump(punch_data, f)


with open(os.path.join(save_test_path, "dataset_punch_targets_" + "right" + ".json"), 'w') as f:
    json.dump(punch_targets_dataset_right.tolist(), f)
xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_right, offset=0)
print("Right target details:")
print("x range : ", xR)
print("y range : ", yR)
print("z range : ", zR)
utils.plot_punch_targets(punch_targets_dataset_right, xR, yR, zR, save_punch_targets_plot_path, "right", "dataset")

with open(os.path.join(save_test_path, "dataset_punch_targets_" + "left" + ".json"), 'w') as f:
    json.dump(punch_targets_dataset_left.tolist(), f)
xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_left, offset=0)
print("Left target details:")
print("x range : ", xR)
print("y range : ", yR)
print("z range : ", zR)

utils.plot_punch_targets(punch_targets_dataset_left, xR, yR, zR, save_punch_targets_plot_path, "left", "dataset")
