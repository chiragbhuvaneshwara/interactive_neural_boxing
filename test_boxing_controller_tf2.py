import json
import os
from src.controlers.boxingControllers.controller_box import BoxingController
from src.nn.keras_mods.mann_keras import MANN as MANNTF
from simple_plotter import simple_matplotlib_plotter
import numpy as np
# from src.controlers.character import *
import pandas as pd

# TODO Write an init README
frd = 1
window = 25
epochs = 20
controller_in_out_dir = 'src/controlers/boxingControllers/controller_in_out'
frd_win_epochs = 'boxing_fr_' + str(frd) + '_' + str(window) + '_' + str(epochs)
trained_base_path = 'trained_models/mann_tf2/' + frd_win_epochs
target_file = os.path.join(trained_base_path, 'model_weights_std_in_out.zip')
# target_file = PureWindowsPath(
#     r"C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\trained_models\mann_tf2\model_20_weights_std_in_out")

# mann_config_path = r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\mann_config_fr_1_20.json'
mann_config_path = os.path.join(trained_base_path, 'mann_config.json')
with open(mann_config_path) as json_file:
    mann_config = json.load(json_file)

mann = MANNTF(mann_config)
mann.load_discrete_weights(target_file)
# dataset_config = "./data/boxing_fr_1_25_config_updated.json"
dataset_config = "data/boxing_fr_"+str(frd)+"_"+str(window)+"/config.json"

with open(dataset_config) as f:
    config_store = json.load(f)

# mann.start_tf()
bc = BoxingController(mann, config_store)

my_plotter = simple_matplotlib_plotter.Plotter()
# right target coordinates, left target coordinates
# Neutral Postion
# target = [0.401022691,	-5.128026,	1.101998588,	0.401022691,	-5.128026,	1.101998588]
# target = [0.302732434,	-5.116577,	1.096424532,	0.302732434,	-5.116577,	1.096424532]
# Left Punch
# target = [0.557932576,	-4.381198,	0.96235335,	-0.216751668,	-1.127927336,	-1.13160189]
target = [0.267201945,	-1.783858,	0.958687371,	-0.212572447,	-1.056275163,	-1.190664312]
# X_csv_path = "data/boxing_fr_"+str(frd)+"_"+str(window)+"/X.csv"
# X_csv = pd.read_csv(X_csv_path)
# reqd_cols = ['x_right_punch_target_0', 'x_right_punch_target_1', 'x_right_punch_target_2', 'x_left_punch_target_0',
#              'x_left_punch_target_1', 'x_left_punch_target_2']
# targets = X_csv[reqd_cols].to_numpy()
in_data_collection = []
out_data_collection = []
poses = []
for f in range(100):
# for f in range(2):
# for f in range(3):
# for target in targets[200:300]:
    ## [start:end] contains both left and right punches in X_csv
    in_data, out_data = bc.pre_render(target)
    in_data_collection.append(np.hstack(in_data))
    out_data_collection.append(np.hstack(out_data))
    poses.append(np.array(bc.char.joint_positions))
    bc.post_render()

X_df = pd.DataFrame(data=in_data_collection, columns=config_store['col_names'][0])
X_df.to_csv(os.path.join(controller_in_out_dir, "X_controller.csv"))
Y_df = pd.DataFrame(data=out_data_collection, columns=config_store['col_names'][1])
Y_df.to_csv(os.path.join(controller_in_out_dir, "Y_controller.csv"))

# Removing unwanted finger joints
# right_unwanted_ids = [i for i in range(17, 36)]
# left_unwanted_ids = [i for i in range(40, 59)]
# indices_unwanted_joints = right_unwanted_ids + left_unwanted_ids
# diff = 36 - 17  # or 59 - 40
# new_left_hand_id = bc.char.hand_left - diff
# new_right_hand_id = bc.char.hand_right
#
# print('Left_id:', new_left_hand_id, 'Right_id:', new_right_hand_id)
# new_poses = []
# for pose in poses:
#     new_pose = np.delete(pose, indices_unwanted_joints, axis=0)
#     new_poses.append(new_pose)
# poses = np.array(new_poses)

print('start')
poses = poses * 10
pos2 = np.array(poses)
my_plotter.animated(pos2[200:300])
print('done')
