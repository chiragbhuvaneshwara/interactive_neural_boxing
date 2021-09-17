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
    foot_gpos_right = []
    foot_gpos_left = []

    for div in range(frame_rate_div):
        print('\n')
        print('Processing Blender csv data %s' % frame_rate_div, div)
        handler.set_awinda_parameters()  # set the skeleton parameters by manually checking the bvh files
        # Loading Bvh file into memory
        handler.load_motion(frame_rate_divisor=frame_rate_div, frame_rate_offset=div)

        gpos = handler.get_glob_pos()
        foot_right = gpos[:, handler.foot_right["t"]]
        foot_left = gpos[:, handler.foot_left["t"]]

        for i in range(handler.window_root, handler.n_frames - handler.window_root - 1, 1):
            if i % 50 == 0:
                print('Frames processed: ', i)
            foot_gpos_right.append(foot_right[i].ravel())
            foot_gpos_left.append(foot_left[i].ravel())

    return np.array(foot_gpos_right), np.array(foot_gpos_left)


def process_folder(bvh_path, punch_labels_path, frame_rate_division, forward_direction, traj_window_root,
                   traj_window_wrist, num_tr_sampling_pts):
    # bvh_files = get_files_in_folder(bvh_path)[:2]
    # punch_labels_files = get_files_in_folder(punch_labels_path)[:2]
    bvh_files = get_files_in_folder(bvh_path)
    punch_labels_files = get_files_in_folder(punch_labels_path)

    foot_pos_per_folder_right = []
    foot_pos_per_folder_left = []

    for b_f, p_f in zip(bvh_files, punch_labels_files):
        print('\n')
        print(b_f, '\n', p_f)
        handler = FeatureExtractor(b_f, traj_window_root, traj_window_wrist, forward_dir=forward_direction,
                                   num_traj_sampling_pts_root=num_tr_sampling_pts)

        foot_pos_cur_file_right, foot_pos_cur_file_left = process_data(handler, p_f, frame_rate_division)

        foot_pos_per_folder_right.append(foot_pos_cur_file_right)
        foot_pos_per_folder_left.append(foot_pos_cur_file_left)

    foot_pos_per_folder_right = np.vstack(foot_pos_per_folder_right)
    foot_pos_per_folder_left = np.vstack(foot_pos_per_folder_left)

    return foot_pos_per_folder_right, foot_pos_per_folder_left


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

foot_pos_dataset_right, foot_pos_dataset_left = process_folder(BVH_PATH, PUNCH_LABELS_PATH,
                                                               FRAME_RATE_DIV,
                                                               FORWARD_DIR,
                                                               TR_WINDOW_ROOT, TR_WINDOW_WRIST,
                                                               TR_SAMPLES)

foot_pos = [foot_pos_dataset_right, foot_pos_dataset_left]
foot_skating = {}
for j, foot in enumerate(["right", "left"]):
    vals = foot_pos[j]
    fs = []
    for i in range(len(vals) - 1):
        curr_disp = ((vals[i + 1, 0] - vals[i, 0]) ** 2 + (vals[i + 1, 2] - vals[i, 2]) ** 2) ** 0.5
        curr_fs = curr_disp * (2 - 2 ** ((vals[i + 1, 1] + vals[i, 1]) / 2 / 0.033))
        fs.append(curr_fs)

    # Appending 0 to ensure fs len is same as col len i.e. assuming displacement 0 for last data point (frame)
    # fs.append(0)
    foot_skating[foot] = np.mean(fs)

print(foot_skating)
