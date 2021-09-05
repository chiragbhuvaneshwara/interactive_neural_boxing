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


eval_df = pd.read_csv("eval.csv")
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

eval_df.to_csv("eval_res.csv")
