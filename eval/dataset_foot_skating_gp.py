import json
import os

import numpy as np


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


def _generate_id_sequence(column_demarcation_ids, key):
    return [i for i in range(column_demarcation_ids[key][0], column_demarcation_ids[key][1])]


data_config_path = os.path.join("data", "neural_data", "root_tr_exp", "fr_1_tr_5_5", "dataset_config.json")
with open(data_config_path) as f:
    dataset_config = json.load(f)

bone_map = dataset_config['bone_map']
col_demarcation_ids = dataset_config['col_demarcation_ids']
x_col_ids = col_demarcation_ids[0]
x_col_demarcation = col_demarcation_ids[0]
tr_win_root = dataset_config["traj_window_root"]
tr_win_wr = dataset_config["traj_window_wrist"]
dataset = dataset_config["dataset_npz_path"]
data = np.load(dataset)
x_train = data["x"]
y_train = data["y"]

f_r_t_id = bone_map["RightToe"] * 3
f_l_t_id = bone_map["LeftToe"] * 3
foot_toes_velocities_ids = [_generate_id_sequence(x_col_ids, 'x_local_pos')[f_r_t_id: f_r_t_id + 3],
                            _generate_id_sequence(x_col_ids, 'x_local_pos')[f_l_t_id: f_l_t_id + 3]]

foot_skating = {}
for j, foot in enumerate(["right", "left"]):
    cols_out = [foot.capitalize() + "Toe_positions" + "_" + str(i) for i in range(3)]
    vals = x_train[:, foot_toes_velocities_ids[j]]
    fs = []
    for i in range(len(vals) - 1):
        curr_disp = ((vals[i + 1, 0] - vals[i, 0]) ** 2 + (vals[i + 1, 2] - vals[i, 2]) ** 2) ** 0.5
        curr_fs = curr_disp * (2 - 2 ** ((vals[i + 1, 1] + vals[i, 1]) / 2 / 0.033))
        fs.append(curr_fs)

    # Appending 0 to ensure fs len is same as col len i.e. assuming displacement 0 for last data point (frame)
    # fs.append(0)
    foot_skating[foot] = np.mean(fs)

# TODO: Compute velocity for stepping
# TODO: Figure out which foot_skating is used in eval and delete either dataset_foot_skating_gp.py or
#  dataset_foot_skating_lp.py
print(foot_skating)
