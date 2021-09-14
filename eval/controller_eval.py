import math
import os

import pandas as pd
import numpy as np


def get_mse_per_frame(df, col_str_1, col_str_2, hand, save=True, per_punch=False):
    out = df.filter(regex=col_str_1).values
    target = df.filter(regex=col_str_2).values
    mean_squared_error = ((out - target) ** 2).mean(axis=1)
    if save and not per_punch:
        df["mse_per_frame" + "_" + hand] = mean_squared_error
    return df, mean_squared_error


def get_l2norm_per_frame(df, col_str_1, col_str_2, hand, save=True, per_punch=False):
    out = df.filter(regex=col_str_1).values
    target = df.filter(regex=col_str_2).values
    l2norm = np.linalg.norm((out - target), axis=1)
    if save and not per_punch:
        df["l2norm_per_frame" + "_" + hand] = l2norm
    return df, l2norm


def get_mse_per_punch(df, hand):
    df = df.copy()
    cols_out = [hand.capitalize() + "Wrist_positions" + "_" + str(i) for i in range(3)]
    cols_target = ["punch_target" + "_" + hand + "_" + str(i) for i in range(3)]
    col_frames = ["punch_frames" + "_" + hand]
    df_per_punch = df.loc[(df['punch_complete' + "_" + hand] == False) &
                          (df['punch_frames' + "_" + hand] != 0),
                          cols_out + cols_target + col_frames]
    start_end = np.diff((np.diff(df_per_punch.index) == 1) + 0, prepend=0, append=0)
    start_idx = np.where(start_end == 1)
    end_idx = np.where(start_end == -1)
    start_idx = [int(i) for i in start_idx[0]]
    end_idx = [int(i) for i in end_idx[0]]

    mean_squared_error_punch = []
    for i in range(len(start_idx)):
        out = df_per_punch.filter(regex=hand.capitalize() + "Wrist_positions").values[start_idx[i]:end_idx[i] + 1]
        target = df_per_punch.filter(regex="punch_target" + "_" + hand).values[start_idx[i]:end_idx[i] + 1]
        mean_squared_error = ((out - target) ** 2).mean()
        mean_squared_error_punch += [mean_squared_error] * (end_idx[i] + 1 - start_idx[i])

    df_per_punch["mse_per_punch" + "_" + hand] = mean_squared_error_punch

    return df_per_punch


def get_punch_accuracy_closest_to_target(df, hand):
    df = df.copy()
    cols_out = [hand.capitalize() + "Wrist_positions" + "_" + str(i) for i in range(3)]
    cols_target = ["punch_target" + "_" + hand + "_" + str(i) for i in range(3)]
    col_mse_per_frame = ["mse_per_frame" + "_" + hand]

    idx_wr_closest_to_target = df.index[df['punch_half_complete' + "_" + hand] == True].tolist()
    idx_wr_closest_to_target = [idx - 1 for idx in idx_wr_closest_to_target]

    df_wr_closest_to_target = df.loc[idx_wr_closest_to_target,
                                     cols_out + cols_target + col_mse_per_frame]

    acc = []
    df_wr_closest_to_target["punch_accuracy_" + hand] = df_wr_closest_to_target[col_mse_per_frame].apply(
        lambda x: True if x < 0.02 else False)

    return df_wr_closest_to_target


def get_foot_skating(df):
    df = df.copy()
    foot_skating = []
    for foot in ["right", "left"]:
        cols_out = [foot.capitalize() + "Toe_positions" + "_" + str(i) for i in range(3)]
        vals = df[cols_out].values
        fs = []
        for i in range(len(vals) - 1):
            curr_disp = ((vals[i + 1, 0] - vals[i, 0]) ** 2 + (vals[i + 1, 2] - vals[i, 2]) ** 2) ** 0.5
            curr_fs = curr_disp * (2 - 2 ** ((vals[i + 1, 1] + vals[i, 1]) / 2 / 0.033))
            fs.append(curr_fs)

        # Appending 0 to ensure fs len is same as col len i.e. assuming displacement 0 for last data point (frame)
        fs.append(0)
        df[foot.capitalize() + "Foot_skating"] = fs
        foot_skating.append(fs)

    return df, fs


def get_path_error(df):
    def _get_proj_x_on_y(x, y):
        proj = y * np.dot(x, y) / np.dot(y, y)
        return proj

    df = df.copy()
    cols_out = ["Hips_positions" + "_" + str(i) for i in range(0, 3, 2)]
    out = df[cols_out].values
    zaxis = np.array([0, 1])

    target = []
    for o in out:
        curr_projection = _get_proj_x_on_y(o, zaxis)
        target.append(curr_projection)

    target = np.array(target)
    curr_pe = ((out - target) ** 2).mean(axis=1)

    df["Path_MSE"] = curr_pe

    return df, curr_pe


frd = 1
window_wrist = math.ceil(5 / frd)
window_root = math.ceil(5 / frd)
epochs = 1000

DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", )
# DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", "dev")

frd_win = 'fr_' + str(frd) + '_tr_' + str(window_root) + "_" + str(window_wrist)
frd_win_epochs = frd_win + '_ep_' + str(epochs)
all_models_path = os.path.join("train", "models", "mann_tf2_v2")
trained_base_path = os.path.join(all_models_path, frd_win_epochs, "2021-09-11_20-00-40")
model_id = trained_base_path.split(os.sep)[-3]

eval_save_path = os.path.join("eval", "saved", "controller")

for model_id in os.listdir(eval_save_path):
    for csv in os.listdir(os.path.join(eval_save_path, model_id, "unity_out")):
        if "punch" in csv:
            eval_df = pd.read_csv(os.path.join(eval_save_path, model_id, "unity_out", csv))
            eval_df, mse_per_frame = get_mse_per_frame(eval_df, "RightWrist_positions", "punch_target_right", "right")
            eval_df, mse_per_frame = get_mse_per_frame(eval_df, "LeftWrist_positions", "punch_target_left", "left")

            eval_df, l2_per_frame = get_l2norm_per_frame(eval_df, "RightWrist_positions", "punch_target_right", "right")
            eval_df, l2_per_frame = get_l2norm_per_frame(eval_df, "LeftWrist_positions", "punch_target_left", "left")

            eval_right_per_punch_df = get_mse_per_punch(eval_df, "right")
            eval_left_per_punch_df = get_mse_per_punch(eval_df, "left")
            eval_df["mse_per_punch" + "_" + "right"] = eval_right_per_punch_df["mse_per_punch" + "_" + "right"]
            eval_df["mse_per_punch" + "_" + "left"] = eval_left_per_punch_df["mse_per_punch" + "_" + "left"]
            eval_df.fillna(0, inplace=True)
            eval_df = eval_df.loc[:, ~eval_df.columns.str.contains('^Unnamed')]

            eval_df["punch_accuracy" + "_" + "right"] = eval_df["l2norm_per_frame" + "_" + "right"].apply(
                lambda x: True if x < 0.02 else False)
            eval_df["punch_accuracy" + "_" + "left"] = eval_df["l2norm_per_frame" + "_" + "left"].apply(
                lambda x: True if x < 0.02 else False)
            print(eval_df)
            print(eval_df.punch_accuracy_right.value_counts())
            print(eval_df.punch_accuracy_left.value_counts())

            res_path = os.path.join(eval_save_path, model_id, "eval_res")
            if not os.path.isdir(res_path):
                os.makedirs(res_path)
            eval_df.to_csv(os.path.join(res_path, csv))

        if "walk" in csv:
            eval_df = pd.read_csv(os.path.join(eval_save_path, model_id, "unity_out", csv))
            eval_df, foot_skating = get_foot_skating(eval_df)
            # print(foot_skating)

            eval_df, path_error = get_path_error(eval_df)
            res_path = os.path.join(eval_save_path, model_id, "eval_res")
            if not os.path.isdir(res_path):
                os.makedirs(res_path)
            eval_df.to_csv(os.path.join(res_path, csv))
