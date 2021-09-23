import glob
import os
import math
import numpy as np
import pandas as pd
import json
import argparse

from data.neural_data_prep.nn_features.processer import process_folder


def setup_output_dir(output_base_path, output_directory):
    """
    Sets up output dir if it doesn't exist. If it does exist, it empties the dir.
    :param output_base_path: str
    :param output_directory: str
    """
    # if output_directory not in os.listdir(output_base_path):
    if not os.path.exists(os.path.join(output_base_path, output_directory)):
        print('Creating new output dir:', os.path.join(output_base_path, output_directory))
        # os.mkdir(os.path.join(output_base_path, output_directory))
        os.makedirs(os.path.join(output_base_path, output_directory))
    else:
        print('Emptying output dir:', os.path.join(output_base_path, output_directory))
        files = glob.glob(os.path.join(output_base_path, output_directory, '*'))
        for fi in files:
            os.remove(fi)


####################### CONTROL PARAMS ###################################
args_parser = argparse.ArgumentParser()
args_parser.add_argument("-d", "--develop", help="Run on subset(",
                         action="store_true", default=False)
args_parser.add_argument("-l", "--local", help="Flag indicating remote machine or local machine",
                         action="store_true", default=False)
args_parser.add_argument("-e", "--exptype", type=int,
                         help="Experiment type: 0 for lr, hidden neurons and gating experts."
                              "1 for trajectory window of root and wrist experiment",
                         required=True)
args_parser.add_argument("-n", "--expname", help="Name for the experiment",
                         type=str, required=True)
args_parser.add_argument("-tr", "--traj_root", nargs="+", type=int)
args_parser.add_argument("-tw", "--traj_wrist", nargs="+", type=int)
args_parser.add_argument("-frd", "--frame_rate_div", type=int, default=1)
args = args_parser.parse_args()

FRAME_RATE_DIV = args.frame_rate_div
ALL_TR_WINS_WRIST = args.traj_wrist if args.traj_wrist else [5]
ALL_TR_WINS_ROOT = args.traj_root if args.traj_root else [5]

ALL_TR_WINS_WRIST = list(set([math.ceil(tr / FRAME_RATE_DIV) for tr in ALL_TR_WINS_WRIST]))
ALL_TR_WINS_ROOT = list(set([math.ceil(tr / FRAME_RATE_DIV) for tr in ALL_TR_WINS_ROOT]))

EXP_NAME = args.expname
EXP_TYPE = args.exptype



DEVELOP = args.develop
LOCAL = args.local

if LOCAL:
    print('Local machine dev')
elif not LOCAL:
    print('Remote machine dev')

INPUT_BASE_PATH = os.path.join("data", "raw_data")

PUNCH_LABELS_PATH = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "tertiary")
BVH_PATH = os.path.join(INPUT_BASE_PATH, "mocap", "hq", "processed")
FORWARD_DIR = np.array([0.0, 0.0, 1.0])

if EXP_TYPE < 0 or EXP_TYPE > 1:
    raise ValueError("Unsupported value: Currently should be 0 or 1")
elif EXP_TYPE == 1:
    print(EXP_NAME)
    print("Root tr wins:", ALL_TR_WINS_ROOT)
    print("Wrist tr wins:", ALL_TR_WINS_WRIST)
    for TR_WINDOW_ROOT in ALL_TR_WINS_ROOT:
        for TR_WINDOW_WRIST in ALL_TR_WINS_WRIST:
            OUTPUT_BASE_PATH = os.path.join("data", "neural_data")

            if DEVELOP:
                print('Dev mode')
                OUTPUT_BASE_PATH = os.path.join(OUTPUT_BASE_PATH, "dev")

            TR_SAMPLES_WRIST = 2 * TR_WINDOW_WRIST
            TR_SAMPLES_ROOT = 2 * TR_WINDOW_ROOT
            ####################### CONTROL PARAMS ###################################

            x_train, y_train, dataset_config = process_folder(BVH_PATH, PUNCH_LABELS_PATH, FRAME_RATE_DIV, FORWARD_DIR,
                                                              TR_WINDOW_ROOT, TR_WINDOW_WRIST, TR_SAMPLES_ROOT,
                                                              TR_SAMPLES_WRIST,
                                                              DEVELOP)
            frd_win = 'fr_' + str(FRAME_RATE_DIV) + '_tr_' + str(TR_WINDOW_ROOT) + "_" + str(TR_WINDOW_WRIST)
            OUTPUT_BASE_PATH = os.path.join(OUTPUT_BASE_PATH, EXP_NAME)
            setup_output_dir(OUTPUT_BASE_PATH, frd_win)
            out_dir = os.path.join(OUTPUT_BASE_PATH, frd_win)

            dataset_npz_path = os.path.join(out_dir, "train")
            np.savez_compressed(dataset_npz_path, x=x_train, y=y_train)
            dataset_config["dataset_npz_path"] = dataset_npz_path + ".npz"
            with open(os.path.join(out_dir, "dataset_config.json"), "w") as f:
                json.dump(dataset_config, f, indent=4)

            if DEVELOP:
                print('X shape is {0} and X mean is {1}'.format(x_train.shape, x_train.mean()))
                print('Y shape is {0} and Y mean is {1}'.format(y_train.shape, y_train.mean()))
                X_train_df = pd.DataFrame(data=x_train, columns=dataset_config['col_names'][0])
                X_train_df.to_csv(os.path.join(out_dir, "x_train_debug.csv"))
                Y_train_df = pd.DataFrame(data=y_train, columns=dataset_config['col_names'][1])
                Y_train_df.to_csv(os.path.join(out_dir, "y_train_debug.csv"))

            print("done")

else:
    TR_WINDOW_WRIST = math.ceil(5 / FRAME_RATE_DIV)
    TR_WINDOW_ROOT = math.ceil(5 / FRAME_RATE_DIV)
    print(EXP_NAME)
    print("Root tr win:", TR_WINDOW_ROOT)
    print("Wrist tr win:", TR_WINDOW_WRIST)

    OUTPUT_BASE_PATH = os.path.join("data", "neural_data")

    if DEVELOP:
        print('Dev mode')
        OUTPUT_BASE_PATH = os.path.join(OUTPUT_BASE_PATH, "dev")

    TR_SAMPLES_WRIST = 2 * TR_WINDOW_WRIST
    TR_SAMPLES_ROOT = 2 * TR_WINDOW_ROOT
    ####################### CONTROL PARAMS ###################################

    x_train, y_train, dataset_config = process_folder(BVH_PATH, PUNCH_LABELS_PATH, FRAME_RATE_DIV, FORWARD_DIR,
                                                      TR_WINDOW_ROOT, TR_WINDOW_WRIST, TR_SAMPLES_ROOT,
                                                      TR_SAMPLES_WRIST,
                                                      DEVELOP)
    frd_win = 'fr_' + str(FRAME_RATE_DIV) + '_tr_' + str(TR_WINDOW_ROOT) + "_" + str(TR_WINDOW_WRIST)
    OUTPUT_BASE_PATH = os.path.join(OUTPUT_BASE_PATH, EXP_NAME)
    setup_output_dir(OUTPUT_BASE_PATH, frd_win)
    out_dir = os.path.join(OUTPUT_BASE_PATH, frd_win)

    dataset_npz_path = os.path.join(out_dir, "train")
    np.savez_compressed(dataset_npz_path, x=x_train, y=y_train)
    dataset_config["dataset_npz_path"] = dataset_npz_path + ".npz"
    with open(os.path.join(out_dir, "dataset_config.json"), "w") as f:
        json.dump(dataset_config, f, indent=4)

    if DEVELOP:
        print('X shape is {0} and X mean is {1}'.format(x_train.shape, x_train.mean()))
        print('Y shape is {0} and Y mean is {1}'.format(y_train.shape, y_train.mean()))
        X_train_df = pd.DataFrame(data=x_train, columns=dataset_config['col_names'][0])
        X_train_df.to_csv(os.path.join(out_dir, "x_train_debug.csv"))
        Y_train_df = pd.DataFrame(data=y_train, columns=dataset_config['col_names'][1])
        Y_train_df.to_csv(os.path.join(out_dir, "y_train_debug.csv"))

    print("done")
