import math
import os
import numpy as np

import pandas as pd

frd = 1
window_wrist = math.ceil(5 / frd)
window_root = math.ceil(5 / frd)
frd_win = 'fr_' + str(frd) + '_tr_' + str(window_root) + "_" + str(window_wrist)

DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", "dev")
# DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data")
dataset_path = os.path.join(DATASET_OUTPUT_BASE_PATH, frd_win, "y_train_debug.csv")
# dataset_path = os.path.join(DATASET_OUTPUT_BASE_PATH, frd_win, "x_train_debug.csv")

df = pd.read_csv(dataset_path)
# print(df)

# pred_fwd_dir_x = df["y_root_velocity_0"]
# pred_fwd_dir_z = df["y_root_velocity_1"]
pred_fwd_dir_x = df["y_root_new_forward_0"]
pred_fwd_dir_z = df["y_root_new_forward_1"]
# pred_fwd_dir_x = df["x_root_fwd_0"]
# pred_fwd_dir_z = df["x_root_fwd_1"]
# pred_fwd_dir_z = df["x_root_fwd_2"]

fwd_dirs = np.array([(pred_fwd_dir_x[i], 0, pred_fwd_dir_z[i]) for i in range(len(pred_fwd_dir_x))])
# fwd_dirs = [(pred_fwd_dir_x[i], pred_fwd_dir_y[i], pred_fwd_dir_z[i]) for i in range(len(pred_fwd_dir_x))]
rads = np.arctan2(pred_fwd_dir_x, pred_fwd_dir_z)
# rotational_vel = np.array([ rads[i] * 180 / math.pi for i in range(len(pred_fwd_dir_x))])
rotational_vel = np.array([ round(math.atan2(pred_fwd_dir_z[i], pred_fwd_dir_x[i]) * 180 / math.pi) for i in range(len(pred_fwd_dir_x))])
# print(rotational_vel)

for i in rotational_vel:
    print(i)

print("*************************")
print(max(rads))
print(min(rads))

# for i in fwd_dirs:
#     # print(np.round(i))
#     print(i)

result = np.all(fwd_dirs == fwd_dirs[0])
if result:
    print('All Values in Array are same / equal')
else:
    print('All Values in Array are not same')

# print("rv", rotational_vel * 180 / math.pi)
