import json
import os
from src.controlers.boxing.controller_tf2_v2 import BoxingController
from src.nn.mann_keras_v2.mann import load_mann, load_binary
from simple_plotter import simple_matplotlib_plotter
import numpy as np
import pandas as pd

# TODO Cleanup
FRD = 1
WINDOW = 15
EPOCHS = 100
CONTROLLER_SAVE_IN_OUT_DIR = 'src/controlers/boxing/controller_in_out'

frd_win_epochs = 'boxing_fr_' + str(FRD) + '_' + str(WINDOW) + '_' + str(EPOCHS)
trained_base_path = 'saved_models/mann_tf2_v2/' + frd_win_epochs + '/20210323_14-51-07/epochs/epoch_98'
target_file = os.path.join(trained_base_path, 'saved_model')

x_mean, y_mean = load_binary(os.path.join(trained_base_path, "means", "Xmean.bin")), \
                 load_binary(os.path.join(trained_base_path, "means", "Ymean.bin"))
x_std, y_std = load_binary(os.path.join(trained_base_path, "means", "Xstd.bin")), \
               load_binary(os.path.join(trained_base_path, "means", "Ystd.bin"))

norm = {
    'x_mean': x_mean,
    'y_mean': y_mean,
    'x_std': x_std,
    'y_std': y_std
}

mann = load_mann(os.path.join(trained_base_path, "saved_model"))

# TODO Rename dataset config file in file system to dataset_config.json instead of config.json
dataset_config = "data/boxing_fr_" + str(FRD) + "_" + str(WINDOW) + "/config.json"

with open(dataset_config) as f:
    config_store = json.load(f)

DATASET_OUTPUT_BASE_PATH = 'data'
frd_win = 'boxing_fr_' + str(FRD) + '_' + str(WINDOW)
dataset_path = os.path.join(DATASET_OUTPUT_BASE_PATH, frd_win, 'train.npz')

bc = BoxingController(mann, config_store, dataset_path, norm)

my_plotter = simple_matplotlib_plotter.Plotter()
# right target coordinates, left target coordinates
# Neutral Postion
label = [0, 0]
target = [0, 0, 0, 0, 0, 0]
# Left Punch
# label = [0, 1]
# target = [0, 0, 0, 0.30000001192092896, 1.6000000238418579, -0.75]
# Right Punch
# target = [0.30000001192092896, 1.6000000238418579, -0.75, 0.0, 0.0, 0.0]

in_data_collection = []
out_data_collection = []
poses = []
for f in range(1000):
    print(f)
    in_data, out_data = bc.pre_render(target, label, space="global")
    # in_data, out_data = bc.pre_render(target, space="local")
    in_data_collection.append(np.hstack(in_data))
    out_data_collection.append(np.hstack(out_data))
    poses.append(np.array(bc.char.joint_positions))
    root_tr, root_vels_tr, right_wr_tr, left_wr_tr, right_wr_vels_tr, left_wrist_vels_tr = bc.get_trajectroy_for_vis()
    # print(f)
    # print(bc.getArmTrajectroy()[0][5])
    # print(bc.getGlobalRoot())
    bc.post_render()

X_df = pd.DataFrame(data=in_data_collection, columns=config_store['col_names'][0])
X_df.to_csv(os.path.join(CONTROLLER_SAVE_IN_OUT_DIR, "X_controller.csv"))
Y_df = pd.DataFrame(data=out_data_collection, columns=config_store['col_names'][1])
Y_df.to_csv(os.path.join(CONTROLLER_SAVE_IN_OUT_DIR, "Y_controller.csv"))



# print('start')
# poses = poses
# pos2 = np.array(poses)
# my_plotter.animated(pos2)
print('done')
