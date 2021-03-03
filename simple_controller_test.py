import json
import os
from src.controlers.boxingControllers.controller_box import BoxingController
from src.nn.keras_mods.mann_keras import MANN as MANNTF
from simple_plotter import simple_matplotlib_plotter
import numpy as np
# from src.controlers.character import *
import pandas as pd

# TODO Cleanup
frd = 1
# window = 25
window = 15
epochs = 100
controller_in_out_dir = 'src/controlers/boxingControllers/controller_in_out'
frd_win_epochs = 'boxing_fr_' + str(frd) + '_' + str(window) + '_' + str(epochs)
trained_base_path = 'trained_models/mann_tf2/' + frd_win_epochs
target_file = os.path.join(trained_base_path, 'model_weights_std_in_out.zip')
mann_config_path = os.path.join(trained_base_path, 'mann_config.json')
with open(mann_config_path) as json_file:
    mann_config = json.load(json_file)

mann = MANNTF(mann_config)
mann.load_discrete_weights(target_file)
dataset_config = "data/boxing_fr_" + str(frd) + "_" + str(window) + "/config.json"

with open(dataset_config) as f:
    config_store = json.load(f)

bc = BoxingController(mann, config_store)

my_plotter = simple_matplotlib_plotter.Plotter()
# right target coordinates, left target coordinates
# Neutral Postion
# target = [0, 0, 0, 0, 0, 0]
# target1 = [0, 0, 0, 0, 0, 0]
# Left Punch
target = [0, 0, 0, -0.347354, 0.578946, -0.521855]
# target = [0, 0, 0, 0.30000001192092896, 1.6000000238418579, -0.75]
# Right Punch
# target = [0.30000001192092896, 1.6000000238418579, -0.75, 0.0, 0.0, 0.0]
# target2 = [-0.30000001192092896, 1.6000000238418579, 0.75, 0.0, 0.0, 0.0]

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
    # if f < 50:
    #     in_data, out_data = bc.pre_render(target1, space="global")
    # else:
    #     in_data, out_data = bc.pre_render(target2, space="global")
    # in_data, out_data = bc.pre_render(target, space="global")
    in_data, out_data = bc.pre_render(target, space="local")
    in_data_collection.append(np.hstack(in_data))
    out_data_collection.append(np.hstack(out_data))
    poses.append(np.array(bc.char.joint_positions))
    # print(f)
    # print(bc.getArmTrajectroy()[0][5])
    # print(bc.getGlobalRoot())
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
# my_plotter.animated(pos2[200:300])
print('done')
