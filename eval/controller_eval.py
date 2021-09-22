import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-whitegrid')


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
    # col_mse_per_frame = ["mse_per_frame" + "_" + hand]
    col_mse_per_frame = ["l2norm_per_frame" + "_" + hand]

    idx_wr_closest_to_target = df.index[(df['punch_half_complete' + "_" + hand] == True)].tolist()

    # subtracting 1 as punch half complete is True for frame after punch is completed
    idx_wr_closest_to_target = [idx - 1 for idx in idx_wr_closest_to_target]

    # df = df.iloc[idx_wr_closest_to_target]
    df_wr_closest_to_target = df.iloc[idx_wr_closest_to_target]
    # df_wr_closest_to_target = df.loc[:, cols_out + cols_target + col_mse_per_frame]

    df_wr_closest_to_target["punch_accuracy_" + hand] = df_wr_closest_to_target[col_mse_per_frame].apply(
        lambda x: x < 0.15)

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


def slice_mse_per_punch(df, hand):
    df = df.copy()
    cols_out = ["mse_per_frame" + "_" + hand, "punch_accuracy" + "_" + hand]
    col_frames = ["punch_frames" + "_" + hand]
    df_per_punch = df.loc[(df['punch_complete' + "_" + hand] == False) &
                          (df['punch_frames' + "_" + hand] != 0),
                          cols_out + col_frames]
    start_end = np.diff((np.diff(df_per_punch.index) == 1) + 0, prepend=0, append=0)
    start_idx = np.where(start_end == 1)
    end_idx = np.where(start_end == -1)
    start_idx = [int(i) for i in start_idx[0]]
    end_idx = [int(i) for i in end_idx[0]]

    sliced_mse_per_punch = []
    punch_accurate = []
    for i in range(len(start_idx)):
        out = df_per_punch["mse_per_frame" + "_" + hand].values[start_idx[i]:end_idx[i] + 1]
        p_acc = df_per_punch["punch_accuracy" + "_" + hand].values[start_idx[i]:end_idx[i] + 1]
        p_acc = True if True in p_acc else False
        sliced_mse_per_punch.append(out)
        punch_accurate.append(p_acc)

    return sliced_mse_per_punch, punch_accurate


def plot_mse_per_punch_slice(eval_df, hand, avg_punch, res_path, csv, plot_avg=True, less_than=True):
    sliced_mse_per_punch, p_acc = slice_mse_per_punch(eval_df, hand)
    fig = plt.figure()
    # ax = plt.axes()
    # plt.figure(figsize=(20, 7))

    if plot_avg and less_than:
        plt.title("MSE levels over average punch duration")
        fname = "less_than_avg_" + csv.split(".")[0] + ".png"

    elif plot_avg and not less_than:
        plt.title("MSE levels over maximum allowed punch duration")
        fname = "greater_than_avg_" + csv.split(".")[0] + ".png"
        plt.axvline(x=avg_punch, label='avg_punch_duration = {}'.format(avg_punch), color="blue", ls='--')
        # place legend outside
        plt.legend()
    else:
        plt.title("MSE levels over all punch durations")
        fname = "all_" + csv.split(".")[0] + ".png"
        plt.axvline(x=avg_punch, label='avg_punch_duration = {}'.format(avg_punch), color="blue", ls='--')
        # place legend outside
        plt.legend()

    plt.xlabel("Frames")
    plt.ylabel("MSE")
    for i, mse_slice in enumerate(sliced_mse_per_punch):
        if p_acc[i] == True:
            color = "green"
        else:
            color = "red"

        if len(mse_slice) <= avg_punch and plot_avg and less_than:
            plt.plot(range(1, len(mse_slice) + 1), mse_slice, color=color)
        elif len(mse_slice) > avg_punch and plot_avg and not less_than:
            plt.plot(range(1, len(mse_slice) + 1), mse_slice, color=color)

        elif not plot_avg and not less_than:
            plt.plot(range(1, len(mse_slice) + 1), mse_slice, color=color)

    plt.savefig(os.path.join(res_path, fname))
    plt.close("all")


# EXP_NAME = "root_tr_exp"
EXP_NAME = "wrist_tr_exp"
eval_save_path = os.path.join("eval", "saved", "controller", EXP_NAME)

AVG_PUNCH_DURATION_DATA = 26  # 26 frames (check data/raw_data/punch_label_gen/analyze/stats.py)

for model_id in os.listdir(eval_save_path):
    if "." not in model_id:
        for csv in os.listdir(os.path.join(eval_save_path, model_id, "unity_out")):
            print("\n", csv)
            if "punch" in csv:
                eval_df = pd.read_csv(os.path.join(eval_save_path, model_id, "unity_out", csv))
                eval_df, mse_per_frame = get_mse_per_frame(eval_df, "RightWrist_positions", "punch_target_right",
                                                           "right")
                eval_df, mse_per_frame = get_mse_per_frame(eval_df, "LeftWrist_positions", "punch_target_left", "left")

                eval_df, l2_per_frame = get_l2norm_per_frame(eval_df, "RightWrist_positions", "punch_target_right",
                                                             "right")
                eval_df, l2_per_frame = get_l2norm_per_frame(eval_df, "LeftWrist_positions", "punch_target_left",
                                                             "left")

                eval_right_per_punch_df = get_mse_per_punch(eval_df, "right")
                eval_left_per_punch_df = get_mse_per_punch(eval_df, "left")
                eval_df["mse_per_punch" + "_" + "right"] = eval_right_per_punch_df["mse_per_punch" + "_" + "right"]
                eval_df["mse_per_punch" + "_" + "left"] = eval_left_per_punch_df["mse_per_punch" + "_" + "left"]
                eval_df.fillna(0, inplace=True)
                eval_df = eval_df.loc[:, ~eval_df.columns.str.contains('^Unnamed')]

                eval_right_p_acc_df = get_punch_accuracy_closest_to_target(eval_df, "right")
                eval_left_p_acc_df = get_punch_accuracy_closest_to_target(eval_df, "left")
                eval_df["punch_accuracy" + "_" + "right"] = eval_right_p_acc_df["punch_accuracy" + "_" + "right"]
                eval_df["punch_accuracy" + "_" + "left"] = eval_left_p_acc_df["punch_accuracy" + "_" + "left"]
                eval_df.fillna(False, inplace=True)
                eval_df = eval_df.loc[:, ~eval_df.columns.str.contains('^Unnamed')]
                # eval_df["punch_accuracy" + "_" + "left"] = eval_df["l2norm_per_frame" + "_" + "left"].apply(
                #     lambda x: True if x < 0.09 else False)

                print(eval_df.punch_accuracy_right.value_counts())
                print(eval_df.punch_accuracy_left.value_counts())
                print("##############")
                res_path = os.path.join(eval_save_path, model_id, "eval_res", "csv")
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)
                eval_df.to_csv(os.path.join(res_path, csv))

                res_path = os.path.join(eval_save_path, model_id, "eval_res", "plots")
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)

                for hand in ["right", "left"]:
                    plot_mse_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, plot_avg=True,
                                             less_than=True)
                    plot_mse_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, plot_avg=True,
                                             less_than=False)
                    plot_mse_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, plot_avg=False,
                                             less_than=False)

                # sliced_mse_per_punch = slice_mse_per_punch(eval_df, "right")
                # fig = plt.figure()
                # ax = plt.axes()
                # plt.title("MSE levels over punch")
                # plt.xlabel("Frames")
                # plt.ylabel("MSE")
                # for mse_slice in sliced_mse_per_punch:
                #     plt.plot(range(1, len(mse_slice) + 1), mse_slice)
                # plt.savefig(os.path.join(res_path, csv.split(".")[0] + ".png"))
                # plt.close(fig)

            if "walk" in csv:
                eval_df = pd.read_csv(os.path.join(eval_save_path, model_id, "unity_out", csv))
                eval_df, foot_skating = get_foot_skating(eval_df)
                # print(foot_skating)

                eval_df, path_error = get_path_error(eval_df)
                res_path = os.path.join(eval_save_path, model_id, "eval_res")
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)
                eval_df.to_csv(os.path.join(res_path, csv))
