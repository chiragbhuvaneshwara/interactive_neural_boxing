import glob
import os
import math
from process_folder import process_folder
import numpy as np
import pandas as pd
import json


# TODO: try to pip install mosi_utils_anim instead of having it in the repo

def setup_output_dir(output_base_path, output_directory):
    """
    Sets up output dir if it doesn't exist. If it does exist, it empties the dir.
    :param output_base_path: str
    :param output_directory: str
    """
    if output_directory not in os.listdir(output_base_path):
        print('Creating new output dir:', output_directory)
        os.mkdir(os.path.join(output_base_path, output_directory))
    else:
        print('Emptying output dir:', output_directory)
        files = glob.glob(os.path.join(output_base_path, output_directory, '*'))
        for fi in files:
            os.remove(fi)


####################### CONTROL PARAMS ###################################
DEVELOP = False
INPUT_BASE_PATH = '../boxing-blender-data-gen'
OUTPUT_BASE_PATH = '../boxing-mosi-dev-vinn/data/'
if DEVELOP:
    INPUT_BASE_PATH = '../VCS-boxing-predictor'
    OUTPUT_BASE_PATH = '../VCS-MOSI-DEV-VINN/mosi_dev_vinn/data/'
    OUTPUT_BASE_PATH += '/dev'
PUNCH_PHASE_PATH = INPUT_BASE_PATH + '/Blender_Code_Snippets/data_annotation_res/new_data/tertiary/'
BVH_PATH = INPUT_BASE_PATH + "/Data/boxing_chirag/hq/processed/"
FRAME_RATE_DIV = 1
FORWARD_DIR = np.array([0.0, 0.0, 1.0])
TR_WINDOW = math.ceil(15 / FRAME_RATE_DIV)
####################### CONTROL PARAMS ###################################

x_train, y_train, Dataset_Config = process_folder(BVH_PATH, PUNCH_PHASE_PATH, FRAME_RATE_DIV, FORWARD_DIR, TR_WINDOW,
                                                  DEVELOP)
frd_win = 'boxing_fr_' + str(FRAME_RATE_DIV) + '_' + str(TR_WINDOW)
setup_output_dir(OUTPUT_BASE_PATH, frd_win)
out_dir = os.path.join(OUTPUT_BASE_PATH, frd_win)

np.savez_compressed(os.path.join(out_dir, "train"), Xun=x_train, Yun=y_train)
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(Dataset_Config, f)

if DEVELOP:
    print('X shape is {0} and X mean is {1}'.format(x_train.shape, x_train.mean()))
    print('Y shape is {0} and Y mean is {1}'.format(y_train.shape, y_train.mean()))
    X_train_df = pd.DataFrame(data=x_train, columns=Dataset_Config['col_names'][0])
    X_train_df.to_csv(os.path.join(out_dir, "x_train_debug.csv"))
    Y_train_df = pd.DataFrame(data=y_train, columns=Dataset_Config['col_names'][1])
    Y_train_df.to_csv(os.path.join(out_dir, "y_train_debug.csv"))

print("done")
