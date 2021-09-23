import argparse
import os
import shutil
from datetime import datetime
import math
from timeit import default_timer as timer

import numpy as np
import random as rn
import tensorflow as tf
from train.utils.process_boxing_data import train_boxing_data

SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-d", "--develop", help="Run on subset",
                         action="store_true", default=False)
args_parser.add_argument("-l", "--local", help="Flag indicating remote machine or local machine",
                         action="store_true", default=False)
args_parser.add_argument("-e", "--exptype", help="Experiment type: 0 for lr, hidden neurons and gating experts."
                                                 "1 for trajectory window of root and wrist experiment",
                         type=int, required=True)
args_parser.add_argument("-n", "--expname", help="Name for the experiment",
                         type=str, required=True)
args_parser.add_argument("-tr", "--traj_root", nargs="+", type=int)
args_parser.add_argument("-tw", "--traj_wrist", nargs="+", type=int)
args_parser.add_argument("-lr", "--learning_rate", nargs="+", type=float)
args_parser.add_argument("-nh", "--num_hidden_neurons", nargs="+", type=int)
args_parser.add_argument("-ng", "--num_gating_experts", nargs="+", type=int)
args_parser.add_argument("-frd", "--frame_rate_div", type=int, default=1)
args = args_parser.parse_args()

FRAME_RATE_DIV = args.frame_rate_div
EXP_TYPE = args.exptype

if EXP_TYPE < 0 or EXP_TYPE > 1:
    raise ValueError("Unsupported value: Currently should be 0 or 1")
elif EXP_TYPE == 1:
    best_lr = 0.001
    best_ng = 8
    best_nh = 512

    # FRAME_RATE_DIV = 1
    ALL_TR_WINS_WRIST = args.traj_wrist if args.traj_wrist else [5]
    ALL_TR_WINS_ROOT = args.traj_root if args.traj_root else [5]

    # ALL_TR_WINS_WRIST = list(set([math.ceil(tr / FRAME_RATE_DIV) for tr in ALL_TR_WINS_WRIST]))
    # ALL_TR_WINS_ROOT = list(set([math.ceil(tr / FRAME_RATE_DIV) for tr in ALL_TR_WINS_ROOT]))
    ALL_TR_WINS_WRIST = list(set(ALL_TR_WINS_WRIST))
    ALL_TR_WINS_ROOT = list(set(ALL_TR_WINS_ROOT))

    EXP_NAME = args.expname

    print(EXP_NAME)
    print("Root tr wins:", ALL_TR_WINS_ROOT)
    print("Wrist tr wins:", ALL_TR_WINS_WRIST)

    DEVELOP = args.develop
    LOCAL = args.local

    # TR_WINDOW_WRIST = math.ceil(5 / FRAME_RATE_DIV)
    # TR_WINDOW_ROOT = math.ceil(5 / FRAME_RATE_DIV)
    OUT_BASE_PATH = os.path.join("train", "models", "mann_tf2_v2")
    OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, EXP_NAME)
    if os.path.exists(OUT_BASE_PATH):
        shutil.rmtree(OUT_BASE_PATH, ignore_errors=False, onerror=None)
        os.mkdir(OUT_BASE_PATH)
    for TR_WINDOW_ROOT in ALL_TR_WINS_ROOT:
        for TR_WINDOW_WRIST in ALL_TR_WINS_WRIST:
            OUT_BASE_PATH = os.path.join("train", "models", "mann_tf2_v2")
            ############################################
            frd_win = 'fr_' + str(FRAME_RATE_DIV) + '_tr_' + str(TR_WINDOW_ROOT) + "_" + str(TR_WINDOW_WRIST)
            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            start_time = timer()

            if not DEVELOP:
                dataset_config_path = os.path.join("data", "neural_data", EXP_NAME, frd_win, "dataset_config.json")
                batch_size = 32
                EPOCHS = 150

            elif DEVELOP:
                print('Dev Mode')
                OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "dev")
                # OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "local_dev_saved_models")
                dataset_config_path = os.path.join("data", "neural_data", "dev", EXP_NAME, frd_win,
                                                   "dataset_config.json")
                batch_size = 2
                EPOCHS = 1
                out_dir = os.path.join(OUT_BASE_PATH)

            frd_win_epochs = frd_win + '_ep_' + str(EPOCHS)

            OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, EXP_NAME)
            out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

            train_boxing_data(dataset_config_path, out_dir, epochs=EPOCHS,
                              batchsize=batch_size, learning_rate=best_lr, num_expert_nodes=best_ng,
                              num_hidden_neurons=best_nh)
            end_time = timer()
            print("Model", current_timestamp, "train time in mins:", round((end_time - start_time) / 60))
else:
    LR = args.learning_rate
    NH = args.num_hidden_neurons
    NG = args.num_gating_experts

    num_none = len([i for i in [LR, NH, NG] if i is None])
    if num_none < 2:
        raise ValueError(
            "You can only conduct exps by varying one of learning rate, number of hidden neurons or number of gating "
            "experts. i.e. cannot vary all of them together")

    EXP_NAME = args.expname
    print(EXP_NAME)
    DEVELOP = args.develop
    LOCAL = args.local

    # FRAME_RATE_DIV = 1
    TR_WINDOW_WRIST = math.ceil(5 / FRAME_RATE_DIV)
    TR_WINDOW_ROOT = math.ceil(5 / FRAME_RATE_DIV)
    OUT_BASE_PATH = os.path.join("train", "models", "mann_tf2_v2")
    OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, EXP_NAME)
    if os.path.exists(OUT_BASE_PATH):
        shutil.rmtree(OUT_BASE_PATH, ignore_errors=False, onerror=None)
        os.mkdir(OUT_BASE_PATH)
    ############################################
    frd_win = 'fr_' + str(FRAME_RATE_DIV) + '_tr_' + str(TR_WINDOW_ROOT) + "_" + str(TR_WINDOW_WRIST)

    if not DEVELOP:
        dataset_config_path = os.path.join("data", "neural_data", EXP_NAME, frd_win, "dataset_config.json")
        # dataset_config_path = os.path.join("data", "neural_data", frd_win, "dataset_config.json")
        batch_size = 32
        EPOCHS = 150

    elif DEVELOP:
        print('Dev Mode')
        OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "dev")
        # OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "local_dev_saved_models")
        if os.path.exists(OUT_BASE_PATH):
            shutil.rmtree(OUT_BASE_PATH, ignore_errors=False, onerror=None)
            os.mkdir(OUT_BASE_PATH)
        dataset_config_path = os.path.join("data", "neural_data", "dev", frd_win, "dataset_config.json")
        batch_size = 2
        EPOCHS = 2
        out_dir = os.path.join(OUT_BASE_PATH)

    frd_win_epochs = frd_win + '_ep_' + str(EPOCHS)

    if LR is not None:
        for lr in LR:
            start_time = timer()
            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

            train_boxing_data(dataset_config_path, out_dir, epochs=EPOCHS, learning_rate=lr,
                              batchsize=batch_size)
            end_time = timer()
            print("Model", current_timestamp, "train time in mins:", round((end_time - start_time) / 60))
    elif NG is not None:
        for ng in NG:
            start_time = timer()
            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

            train_boxing_data(dataset_config_path, out_dir, epochs=EPOCHS, num_expert_nodes=ng,
                              batchsize=batch_size)
            end_time = timer()
            print("Model", current_timestamp, "train time in mins:", round((end_time - start_time) / 60))
    else:
        for nh in NH:
            start_time = timer()
            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

            train_boxing_data(dataset_config_path, out_dir, epochs=EPOCHS, num_hidden_neurons=nh,
                              batchsize=batch_size)
            end_time = timer()
            print("Model", current_timestamp, "train time in mins:", round((end_time - start_time) / 60))
