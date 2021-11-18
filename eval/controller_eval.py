import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-whitegrid')


# TODO: move functions to separate file
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


def get_avg_mse_punch_closest_to_target(df, hand):
    df = df.copy()
    mse_at_targets_df = df.loc[
        (df['punch_complete' + "_" + hand] == True), ['mse_per_frame_' + hand, 'punch_complete' + "_" + hand]]
    avg_mse_at_targets = np.mean(mse_at_targets_df['mse_per_frame_' + hand])

    return avg_mse_at_targets


def get_punch_accuracy_closest_to_target(df, hand, p_acc_threshold):
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

    # TODO: Compare with punch accuracy 0.15, 0.3 and 0.45 ==> 0.15m is average size of human head
    df_wr_closest_to_target["punch_accuracy_" + hand] = df_wr_closest_to_target[col_mse_per_frame].apply(
        lambda x: x < p_acc_threshold)

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


            # curr_fs = curr_disp * (2 - 2 ** ((vals[i + 1, 1] + vals[i, 1]) / 2 / 0.033))
            curr_fs = curr_disp * (2 - 2 ** ((vals[i + 1, 1] + vals[i, 1]) / 2 / HEIGHT_THRESHOLD))
            # if CONVERT_METRICS_TO_CENTIMETRE:
            #     curr_fs *= 100
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


def slice_l2norm_per_punch(df, hand):
    df = df.copy()
    cols_out = ["l2norm_per_frame" + "_" + hand, "punch_accuracy" + "_" + hand]
    col_frames = ["punch_frames" + "_" + hand]
    df_per_punch = df.loc[(df['punch_complete' + "_" + hand] == False) &
                          (df['punch_frames' + "_" + hand] != 0),
                          cols_out + col_frames]
    start_end = np.diff((np.diff(df_per_punch.index) == 1) + 0, prepend=0, append=0)
    start_idx = np.where(start_end == 1)
    end_idx = np.where(start_end == -1)
    start_idx = [int(i) for i in start_idx[0]]
    end_idx = [int(i) for i in end_idx[0]]

    sliced_l2_per_punch = []
    punch_accurate = []
    for i in range(len(start_idx)):
        out = df_per_punch["l2norm_per_frame" + "_" + hand].values[start_idx[i]:end_idx[i] + 1]
        p_acc = df_per_punch["punch_accuracy" + "_" + hand].values[start_idx[i]:end_idx[i] + 1]
        p_acc = True if True in p_acc else False
        sliced_l2_per_punch.append(out)
        punch_accurate.append(p_acc)

    return sliced_l2_per_punch, punch_accurate


def plot_mse_per_punch_slice(eval_df, hand, avg_punch, res_path, csv, plot_avg=True, less_than=True):
    sliced_mse_per_punch, p_acc = slice_mse_per_punch(eval_df, hand)
    fig = plt.figure()
    # ax = plt.axes()
    # plt.figure(figsize=(20, 7))

    if plot_avg and less_than:
        plt.title("MSE levels over average punch duration")
        fname = "mse_less_than_avg_" + csv.split(".")[0] + ".png"

    elif plot_avg and not less_than:
        plt.title("MSE levels over maximum allowed punch duration")
        fname = "mse_greater_than_avg_" + csv.split(".")[0] + ".png"
        plt.axvline(x=avg_punch, label='avg_punch_duration = {}'.format(avg_punch), color="blue", ls='--')
        # place legend outside
        plt.legend()
    else:
        plt.title("MSE levels over all punch durations")
        fname = "mse_all_" + csv.split(".")[0] + ".png"
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


def plot_l2_per_punch_slice(eval_df, hand, avg_punch, res_path, csv, p_acc_threshold, plot_avg=True, less_than=True):
    sliced_mse_per_punch, p_acc = slice_l2norm_per_punch(eval_df, hand)
    fig = plt.figure()
    # ax = plt.axes()
    # plt.figure(figsize=(20, 7))

    if plot_avg and less_than:
        plt.title("L2 norm levels over average punch duration")
        fname = "l2_less_than_avg_" + csv.split(".")[0] + ".png"

    elif plot_avg and not less_than:
        plt.title("L2 norm over maximum allowed punch duration")
        fname = "l2_greater_than_avg_" + csv.split(".")[0] + ".png"
        # plt.axvline(x=avg_punch, label='avg_punch_duration = {}'.format(avg_punch), color="blue", ls='--')
        # # place legend outside
        # plt.legend()
    else:
        plt.title("L2 norm over all punch durations")
        fname = "l2_all_" + csv.split(".")[0] + ".png"
        # plt.axvline(x=avg_punch, label='avg_punch_duration = {}'.format(avg_punch), color="blue", ls='--')
        # plt.axhline(y=p_acc_threshold, label='punch accuracy threshold = {}'.format(p_acc_threshold), color="yellow",
        #             ls='--')
        # # place legend outside
        # plt.legend()

    plt.xlabel("Frames")
    plt.ylabel("L2 norm")
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

    plt.axvline(x=avg_punch, label='avg punch duration = {}'.format(avg_punch), color="blue", ls='--', linewidth=4)
    plt.axhline(y=p_acc_threshold, label='punch accuracy threshold = {}'.format(p_acc_threshold), color="black",
                ls=':', linewidth=4)
    # place legend outside
    plt.legend(loc="upper right", fancybox=True, frameon=True, framealpha=1.0)
    # plt.legend(bbox_to_anchor=(0.75, 0.75))

    plt.savefig(os.path.join(res_path, fname))
    plt.close("all")


# TODO: Corrrect path error in plot by projecting points onto Z axis and blending them
def plot_path_error(path_error, res_path):
    fig = plt.figure()
    title_addon = ""
    if "forward" in csv:
        title_addon = "forward"
    elif "backward" in csv:
        title_addon = "backward"
    plt.title("MSE path error over full test " + title_addon + " walk duration")
    fname = "walk_path_error_" + csv.split(".")[0] + ".png"
    plt.axhline(y=round(np.mean(path_error), 4), label='Avg Path Error = {}'.format(round(np.mean(path_error), 4)),
                color="blue", ls='--')
    # place legend outside
    plt.legend()

    plt.xlabel("Time in seconds")
    plt.ylabel("MSE path error")
    plt.plot(range(1, len(path_error) + 1), path_error)

    plt.savefig(os.path.join(res_path, fname))
    plt.close("all")


# EXP_NAME = "root_tr_exp_fr_2"  # 10 models
# EXP_NAME = "root_tr_exp_fr_3"  # 1 models
EXP_NAME = "root_tr_exp_fr_4"  # 2 models
# EXP_NAME = "wrist_tr_exp_fr_1"
# EXP_NAME = "wrist_tr_exp_fr_2"
# EXP_NAME = "wrist_tr_exp_fr_3"
eval_save_path = os.path.join("eval", "saved", "controller", EXP_NAME)

CONVERT_METRICS_TO_CENTIMETRE = True

AVG_PUNCH_DURATION_DATA = 26  # 26 frames (check data/raw_data/punch_label_gen/analyze/stats.py)
PUNCH_ACC_THRESHOLD = 0.15
HEIGHT_THRESHOLD = 0.033
if CONVERT_METRICS_TO_CENTIMETRE:
    PUNCH_ACC_THRESHOLD *= 100
    HEIGHT_THRESHOLD *= 100
punch_summary = {}
walk_summary = {}


def convert_df_to_cm(df):
    # df_cm = df.copy()
    m_cols_bool_arr = df.columns.str.contains("pos") | df.columns.str.contains("vels") | df.columns.str.contains(
        "target")
    all_cols = np.array(df.columns)
    m_col_names = [all_cols[i] for i in range(len(all_cols)) if m_cols_bool_arr[i] == True]
    df[m_col_names] = df[m_col_names] * 100
    return df


for model_id in sorted(os.listdir(eval_save_path)):

    if "." not in model_id:
        for csv in sorted(os.listdir(os.path.join(eval_save_path, model_id, "unity_out"))):
            print("\n", model_id, csv)
            # all_punch_average_mse = {}
            # all_punch_accuracy = {}
            # all_walk_average_path_error = {}
            # all_walk_average_foot_skating = {}

            if "punch" in csv:
                eval_df = pd.read_csv(os.path.join(eval_save_path, model_id, "unity_out", csv), index_col=0)
                if CONVERT_METRICS_TO_CENTIMETRE:
                    eval_df = convert_df_to_cm(eval_df)
                eval_df, foot_skating = get_foot_skating(eval_df)
                eval_df, mse_per_frame = get_mse_per_frame(eval_df, "RightWrist_positions", "punch_target_right",
                                                           "right")
                eval_df, mse_per_frame = get_mse_per_frame(eval_df, "LeftWrist_positions", "punch_target_left", "left")

                eval_df, l2_per_frame = get_l2norm_per_frame(eval_df, "RightWrist_positions", "punch_target_right",
                                                             "right")
                eval_df, l2_per_frame = get_l2norm_per_frame(eval_df, "LeftWrist_positions", "punch_target_left",
                                                             "left")

                # eval_right_per_punch_df = get_mse_per_punch(eval_df, "right")
                # eval_left_per_punch_df = get_mse_per_punch(eval_df, "left")
                # eval_df["mse_per_punch" + "_" + "right"] = eval_right_per_punch_df["mse_per_punch" + "_" + "right"]
                # eval_df["mse_per_punch" + "_" + "left"] = eval_left_per_punch_df["mse_per_punch" + "_" + "left"]
                eval_df.fillna(0, inplace=True)
                eval_df = eval_df.loc[:, ~eval_df.columns.str.contains('^Unnamed')]

                eval_right_p_acc_df = get_punch_accuracy_closest_to_target(eval_df, "right", PUNCH_ACC_THRESHOLD)
                eval_left_p_acc_df = get_punch_accuracy_closest_to_target(eval_df, "left", PUNCH_ACC_THRESHOLD)
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

                punch_accuracy = []
                average_mse = []
                for hand in ["right", "left"]:
                    plot_mse_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, plot_avg=True,
                                             less_than=True)
                    plot_mse_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, plot_avg=True,
                                             less_than=False)
                    plot_mse_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, plot_avg=False,
                                             less_than=False)

                    plot_l2_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, PUNCH_ACC_THRESHOLD,
                                            plot_avg=True,
                                            less_than=True)
                    plot_l2_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, PUNCH_ACC_THRESHOLD,
                                            plot_avg=True,
                                            less_than=False)
                    plot_l2_per_punch_slice(eval_df, hand, AVG_PUNCH_DURATION_DATA, res_path, csv, PUNCH_ACC_THRESHOLD,
                                            plot_avg=False,
                                            less_than=False)

                    total_num_punches_per_hand = int(csv.split("_")[-1].split(".")[0]) / 2
                    sliced_mse_per_punch, p_acc_per_frame = slice_mse_per_punch(eval_df, hand)

                    # avg_mse = np.mean([np.mean(a) for a in sliced_mse_per_punch])
                    avg_mse = get_avg_mse_punch_closest_to_target(eval_df, hand)
                    average_mse.append(avg_mse)

                    p_acc = (np.sum(p_acc_per_frame) / total_num_punches_per_hand) * 100
                    punch_accuracy.append(p_acc)

                # appending overall average
                average_mse.append(np.mean(average_mse))
                punch_accuracy.append(np.mean(punch_accuracy))
                # TODO compute average or median of MSE when punches are accurate
                # TODO compute average foot skating during punching
                avg_foot_skating = np.mean(foot_skating)

                wrist_tr = int(model_id.split("_ep")[0].split("_tr_5_")[1]) * 2
                frame_skip = int(model_id.split("fr_")[1].split("_tr_")[0])
                # wrist_tr = wrist_tr * frame_skip

                punch_summary[model_id] = [wrist_tr, frame_skip] + average_mse + punch_accuracy + [avg_foot_skating]

            elif "walk" in csv:
                # TODO: Compute velocity for stepping
                # TODO: Compute distance covered during entire walk test
                eval_df = pd.read_csv(os.path.join(eval_save_path, model_id, "unity_out", csv), index_col=0)
                if CONVERT_METRICS_TO_CENTIMETRE:
                    eval_df = convert_df_to_cm(eval_df)
                eval_df, foot_skating = get_foot_skating(eval_df)
                # print(foot_skating)

                eval_df, path_error = get_path_error(eval_df)
                res_path = os.path.join(eval_save_path, model_id, "eval_res", "csv")
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)
                eval_df.to_csv(os.path.join(res_path, csv))

                res_path = os.path.join(eval_save_path, model_id, "eval_res", "plots")
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)

                plot_path_error(path_error, res_path)

                avg_foot_skating = np.mean(foot_skating)
                avg_path_error = np.mean(path_error)

                root_tr = int(model_id.split("_5_ep")[0].split("_tr_")[1]) * 2
                frame_skip = int(model_id.split("fr_")[1].split("_tr_")[0])
                # root_tr = root_tr * frame_skip

                walk_summary[model_id] = [root_tr, frame_skip, avg_path_error, avg_foot_skating]

if CONVERT_METRICS_TO_CENTIMETRE:
    print("################################")
    print("Converting metrics from m to cm:")
    print("################################")


if len(punch_summary.values()) > 0:
    punch_summary_df = pd.DataFrame.from_dict(punch_summary, orient='index',
                                              columns=["Window size", "Frame skip",
                                                       "Avg MSE at target right", "Avg MSE at target left",
                                                       "Avg MSE at target overall",
                                                       "Accuracy right", "Accuracy left", "Accuracy overall",
                                                       "Average Foot Skating"])
    punch_summary_df.to_csv(os.path.join(eval_save_path, "punch_summary.csv"))

    print(punch_summary_df.round({
        "Avg MSE at target right": 2,
        "Avg MSE at target left": 2,
        "Avg MSE at target overall": 2,
        "Accuracy right": 2,
        "Accuracy left": 2,
        "Accuracy overall": 2,
        "Average Foot Skating": 3
    }).to_latex(index=False))
if len(walk_summary.values()) > 0:
    walk_summary_df = pd.DataFrame.from_dict(walk_summary, orient='index',
                                             columns=["Window size", "Frame skip", "Avg path error",
                                                      "Avg foot skating"])
    walk_summary_df.to_csv(os.path.join(eval_save_path, "walk_summary.csv"))
    print(walk_summary_df.round({
        "Avg path error": 2,
        "Avg foot skating": 3
    }).to_latex(index=False))

# TODO either save dfs as latex here or create new script that generates the latex directly by looking through all
#  folders called "exp_x"
