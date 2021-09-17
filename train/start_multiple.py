import argparse
import os
import shutil
from datetime import datetime
import math
from timeit import default_timer as timer

from train.utils.process_boxing_data import train_boxing_data

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-d", "--develop", help="Run on subset",
                         action="store_true", default=False)
args_parser.add_argument("-l", "--local", help="Flag indicating remote machine or local machine",
                         action="store_true", default=False)
args_parser.add_argument("-n", "--expname", help="Name for the experiment",
                         type=str, required=True)
args_parser.add_argument("-tr", "--traj_root", nargs="+", type=int)
args_parser.add_argument("-tw", "--traj_wrist", nargs="+", type=int)
args = args_parser.parse_args()

FRAME_RATE_DIV = 1
ALL_TR_WINS_WRIST = args.traj_wrist if args.traj_wrist else [5]
ALL_TR_WINS_ROOT = args.traj_root if args.traj_root else [5]

ALL_TR_WINS_WRIST = list(set([math.ceil(tr / FRAME_RATE_DIV) for tr in ALL_TR_WINS_WRIST]))
ALL_TR_WINS_ROOT = list(set([math.ceil(tr / FRAME_RATE_DIV) for tr in ALL_TR_WINS_ROOT]))

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
            dataset_config_path = os.path.join("data", "neural_data", "dev", EXP_NAME, frd_win, "dataset_config.json")
            batch_size = 2
            EPOCHS = 1
            out_dir = os.path.join(OUT_BASE_PATH)

        frd_win_epochs = frd_win + '_ep_' + str(EPOCHS)

        OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, EXP_NAME)
        out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

        train_boxing_data(dataset_config_path, out_dir, epochs=EPOCHS,
                          batchsize=batch_size)
        end_time = timer()
        print("Model", current_timestamp,"train time in mins:", round((end_time - start_time) / 60))
