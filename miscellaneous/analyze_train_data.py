import math
import os

import pandas as pd

frd = 1
window_wrist = math.ceil(5 / frd)
window_root = math.ceil(5 / frd)
frd_win = 'fr_' + str(frd) + '_tr_' + str(window_root) + "_" + str(window_wrist)

DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", "dev")
dataset_path = os.path.join(DATASET_OUTPUT_BASE_PATH, frd_win, "y_train_debug.csv")

df = pd.read_csv(dataset_path)
# print(df)

pred_fwd_dir_x = df["y_root_velocity_0"]
pred_fwd_dir_z = df["y_root_velocity_1"]

fwd_dirs = [(pred_fwd_dir_x[i], 0, pred_fwd_dir_z[i]) for i in range(len(pred_fwd_dir_x))]
rotational_vel = [math.atan2(pred_fwd_dir_x[i], pred_fwd_dir_z[i]) * 180 / math.pi for i in range(len(pred_fwd_dir_x))]
# print(rotational_vel)

# for i in rotational_vel:
#     print(i)

for i in fwd_dirs:
    print(i)

# print("rv", rotational_vel * 180 / math.pi)
