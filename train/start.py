import argparse
import os
import shutil
from datetime import datetime
import math
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
args_parser.add_argument("-frd", "--frame_rate_div", type=int, default=1)
args = args_parser.parse_args()

FRAME_RATE_DIV = args.frame_rate_div
DEVELOP = args.develop
LOCAL = args.local

# FRAME_RATE_DIV = 1

TR_WINDOW_WRIST = math.ceil(5 / FRAME_RATE_DIV)
TR_WINDOW_ROOT = math.ceil(5 / FRAME_RATE_DIV)
OUT_BASE_PATH = os.path.join("train", "models", "mann_tf2_v2")
############################################
frd_win = 'fr_' + str(FRAME_RATE_DIV) + '_tr_' + str(TR_WINDOW_ROOT) + "_" + str(TR_WINDOW_WRIST)
current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not DEVELOP:
    dataset_config_path = os.path.join("data", "neural_data", frd_win, "dataset_config.json")
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
    EPOCHS = 1
    out_dir = os.path.join(OUT_BASE_PATH)

frd_win_epochs = frd_win + '_ep_' + str(EPOCHS)
out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

train_boxing_data(dataset_config_path, out_dir, epochs=EPOCHS,
                  batchsize=batch_size)
