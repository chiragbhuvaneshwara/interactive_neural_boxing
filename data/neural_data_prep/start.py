import glob
import os
import math
import numpy as np
import pandas as pd
import json
import argparse
from data.neural_data_prep.process_folder import process_folder


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
args_parser = argparse.ArgumentParser()
args_parser.add_argument("-d", "--develop", help="Run on subset",
                         action="store_true", default=False)
args_parser.add_argument("-l", "--local", help="Flag indicating remote machine or local machine",
                         action="store_true", default=False)
args = args_parser.parse_args()
DEVELOP = args.develop
LOCAL = args.local

if LOCAL:
    print('Local machine dev')
elif not LOCAL:
    print('Remote machine dev')

INPUT_BASE_PATH = os.path.join("data", "raw_data")
OUTPUT_BASE_PATH = os.path.join("data", "neural_data")

if DEVELOP:
    print('Dev mode')
    OUTPUT_BASE_PATH += '/dev'

PUNCH_PHASE_PATH = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "tertiary")
BVH_PATH = os.path.join(INPUT_BASE_PATH, "mocap", "hq", "processed")
FRAME_RATE_DIV = 1
FORWARD_DIR = np.array([0.0, 0.0, 1.0])
TR_WINDOW = math.ceil(15 / FRAME_RATE_DIV)
####################### CONTROL PARAMS ###################################

x_train, y_train, dataset_config = process_folder(BVH_PATH, PUNCH_PHASE_PATH, FRAME_RATE_DIV, FORWARD_DIR, TR_WINDOW,
                                                  DEVELOP)
frd_win = 'boxing_fr_' + str(FRAME_RATE_DIV) + '_' + str(TR_WINDOW)
setup_output_dir(OUTPUT_BASE_PATH, frd_win)
out_dir = os.path.join(OUTPUT_BASE_PATH, frd_win)

np.savez_compressed(os.path.join(out_dir, "train"), x=x_train, y=y_train)
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(dataset_config, f, indent=4)

if DEVELOP:
    print('X shape is {0} and X mean is {1}'.format(x_train.shape, x_train.mean()))
    print('Y shape is {0} and Y mean is {1}'.format(y_train.shape, y_train.mean()))
    X_train_df = pd.DataFrame(data=x_train, columns=dataset_config['col_names'][0])
    X_train_df.to_csv(os.path.join(out_dir, "x_train_debug.csv"))
    Y_train_df = pd.DataFrame(data=y_train, columns=dataset_config['col_names'][1])
    Y_train_df.to_csv(os.path.join(out_dir, "y_train_debug.csv"))

print("done")
