import pandas as pd
import numpy as np

def get_mse_per_frame(df, col_str_1, col_str_2, hand, save=True, per_punch=False):
    out = df.filter(regex=col_str_1).values
    target = df.filter(regex=col_str_2).values
    mean_squared_error = ((out - target) ** 2).mean(axis=1)
    if save and not per_punch:
        df["mse_per_frame" + "_" + hand] = mean_squared_error
    if save and per_punch:
        df["mse_per_punch" + "_" + hand] = mean_squared_error
    return df, mean_squared_error


eval_df = pd.read_csv("eval.csv")
eval_df, mse_per_frame = get_mse_per_frame(eval_df, "RightWrist_positions", "punch_target_right", "right")
eval_df, mse_per_frame = get_mse_per_frame(eval_df, "LeftWrist_positions", "punch_target_left", "left")
# eval_df.to_csv("eval.csv")
print(eval_df)


def get_mse_per_punch(df, hand):
    cols_out = [hand.capitalize() + "Wrist_positions" + "_" + str(i) for i in range(3)]
    cols_target = ["punch_target" + "_" + hand + "_" + str(i) for i in range(3)]
    # _, mse_per_punch = get_mse_per_frame(
    #     df.loc[(df['punch_complete' + "_" + hand] == False) &
    #            (df['punch_frames' + "_" + hand] != 0),
    #            cols_out + cols_target],
    #     "RightWrist_positions",
    #     "punch_target_right",
    #     "right", per_punch=True)
    df_right_punch = df.loc[(df['punch_complete' + "_" + hand] == False) &
                (df['punch_frames' + "_" + hand] != 0),
                cols_out + cols_target]
    start_end = np.diff((np.diff(df_right_punch.index) == 1) + 0, prepend=0, append=0)
    start_idx = np.where(start_end == 1)
    end_idx = np.where(start_end == -1)
    start_idx = [int(i) for i in start_idx[0]]
    end_idx = [int(i) for i in end_idx[0]]

    mean_squared_error_punch = []
    for i in range(len(start_idx)):
        out = df_right_punch.filter(regex=hand.capitalize() + "Wrist_positions").values[start_idx[i]:end_idx[i]]
        target = df_right_punch.filter(regex="punch_target" + "_" + hand).values[start_idx[i]:end_idx[i]]
        mean_squared_error = ((out - target) ** 2).mean()
        mean_squared_error_punch += [mean_squared_error]*(end_idx[i] * start_idx[i])

    # mean_squared_error_punch = pd.Series([mean_squared_error] * len(df.index))
    mean_squared_error_punch = pd.Series(mean_squared_error_punch)
    df_right_punch["mse_per_punch" + "_" + hand] = mean_squared_error_punch

    return df_right_punch


df = eval_df
col_str_1 = ["RightWrist_positions" + "_" + str(i) for i in range(3)]
col_str_2 = ["punch_target_right" + "_" + str(i) for i in range(3)]
_, mse_per_punch = get_mse_per_frame(
    df.loc[(df['punch_complete_right'] == False) & (df['punch_frames_right'] != 0), col_str_1 + col_str_2],
    "RightWrist_positions", "punch_target_right", "right", per_punch=True)
# df["mse_per_punch" + "_" + "right"] = _["mse_per_punch_right"]
df["mse_per_punch" + "_" + "right"] = get_mse_per_punch(df, "right")["mse_per_punch" + "_" + "right"]
df.fillna(0, inplace=True)
df.to_csv("eval_res.csv")
