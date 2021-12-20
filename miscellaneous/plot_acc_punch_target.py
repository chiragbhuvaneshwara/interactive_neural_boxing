import pandas as pd
import json
import os

from eval.gen_punch_targets.utils.cube_grid import cube_grid_points, cube_grid_count
import numpy as np
import eval.gen_punch_targets.utils.common as utils

# results for tr with window size 14 and frameskip 3
eval_df = pd.read_csv(
    "eval/saved/controller/wrist_tr_exp_fr_3/fr_3_tr_5_7_ep_150_2021-09-24_01-24-00/df_with_acc_col_w_14_fs_3.csv")

accurate_targets = {}
for hand in ["left", "right"]:
    cols_out = ["punch_target_" + hand + "_" + str(i) for i in range(3)]
    cols_acc = ['punch_accuracy' + "_" + hand]
    df1 = eval_df.loc[(eval_df['punch_accuracy' + "_" + hand] == True),
                      cols_out + cols_acc]
    df2 = eval_df.loc[(eval_df['punch_half_complete' + "_" + hand] == True),
                      cols_out + cols_acc]
    idxs = [i - 1 for i in df2.index.to_list()]

    df2 = eval_df.loc[idxs, cols_out + cols_acc]
    df3 = df2.merge(df1, how="outer", on=cols_out)
    df2[cols_out] = df2[cols_out].apply(lambda x: x * 0.01)

    colors = df2['punch_accuracy' + "_" + hand].apply(lambda x: [1, 0, 0] if x is False else [0, 1, 0])
    transparency = df2['punch_accuracy' + "_" + hand].apply(lambda x: 15 if x is False else 15)

    accurate_targets[hand] = {
        "targets": df2[cols_out].values,
        "colors": list(colors),
        "transparency": list(transparency)
    }


def gen_uniform_punch_targets(x_range, y_range, z_range, save_path, hand):
    xv = utils.setup_cube_end_pts(x_range, y_range, z_range)
    punch_targets = cube_grid_points(NS, xv, NG).tolist()

    with open(os.path.join(save_path, "uniform_punch_targets_" + hand + ".json"), 'w') as f:
        json.dump(punch_targets, f)

    return punch_targets


NUM_POINTS = 5 ** 3
NS = np.zeros(3, dtype=np.int32);
NS[0] = NUM_POINTS ** (1 / 3) - 1
NS[1] = NUM_POINTS ** (1 / 3) - 1
NS[2] = NUM_POINTS ** (1 / 3) - 1

OFFSET = 0

NG = cube_grid_count(NS)
print('')
print('  Number of grid points will be %d' % (NG))

OFFSET = 0

OUTPUT_BASE_PATH = os.path.join("eval", "saved")
saved_punch_targets_json = os.path.join(OUTPUT_BASE_PATH, "targets", "data",
                                        "dataset_punch_targets_local_pos_py_space.json")

uniform_targets_plots_save_path = os.path.join("miscellaneous/target_acc_plots")
uniform_targets_save_path = os.path.join("miscellaneous/target_acc_plots")

with open(saved_punch_targets_json) as json_file:
    punch_data = json.load(json_file)

punch_targets_dataset_right = np.array(punch_data["punch_targets_dataset_right"])
punch_targets_dataset_left = np.array(punch_data["punch_targets_dataset_left"])

xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_left, OFFSET)
d = accurate_targets["left"]
uniform_punch_targets = gen_uniform_punch_targets(xR, yR, zR, uniform_targets_save_path, "left")

# print(np.sort(np.array(uniform_punch_targets))- np.sort(np.array(d["targets"])))

utils.plot_colored_punch_targets(uniform_punch_targets, d["colors"], d["transparency"], xR, yR, zR,
                                 uniform_targets_plots_save_path, "left", "uniform")

xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_right, OFFSET)
d = accurate_targets["right"]
d["colors"] = d["colors"] + [[1, 0, 0],[1, 0, 0], [1, 0, 0], [1, 0, 0]]
d["transparency"] = d["transparency"] + [5, 5, 5, 5]
uniform_punch_targets = gen_uniform_punch_targets(xR, yR, zR, uniform_targets_save_path, "right")
utils.plot_colored_punch_targets(uniform_punch_targets, d["colors"], d["transparency"], xR, yR, zR,
                                 uniform_targets_plots_save_path, "right", "uniform")
