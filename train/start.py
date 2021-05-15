import argparse
import os
import shutil
from datetime import datetime

from train.utils.process_boxing_data import train_boxing_data

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-d", "--develop", help="Run on subset",
                         action="store_true", default=False)
args_parser.add_argument("-l", "--local", help="Flag indicating remote machine or local machine",
                         action="store_true", default=False)
args = args_parser.parse_args()
DEVELOP = args.develop
LOCAL = args.local

EPOCHS = 100
FRD = 1
WINDOW = 15
OUT_BASE_PATH = os.path.join("train", "models", "mann_tf2_v2")
############################################
frd_win = 'boxing_fr_' + str(FRD) + '_' + str(WINDOW)
current_timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

if not DEVELOP:
    dataset_config_path = os.path.join("data", "neural_data", frd_win, "config.json")
    dataset_npz_path = os.path.join("data", "neural_data", frd_win, "train.npz")
    batch_size = 32

elif DEVELOP:
    print('Dev Mode')
    OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "dev")
    dataset_config_path = os.path.join("data", "neural_data", "dev", frd_win, "config.json")
    dataset_npz_path = os.path.join("data", "neural_data", "dev", frd_win, "train.npz")
    batch_size = 2
    EPOCHS = 2

frd_win_epochs = frd_win + '_' + str(EPOCHS)
if LOCAL:
    print('Local machine dev')
    OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "local_dev_saved_models")
    if os.path.exists(OUT_BASE_PATH):
        shutil.rmtree(OUT_BASE_PATH, ignore_errors=False, onerror=None)
        os.mkdir(OUT_BASE_PATH)
    dataset_config_path = os.path.join("data", "neural_data", "dev", frd_win, "config.json")
    dataset_npz_path = os.path.join("data", "neural_data", "dev", frd_win, "train.npz")
    batch_size = 2
    EPOCHS = 2
    out_dir = os.path.join(OUT_BASE_PATH)

elif not LOCAL:
    out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)

train_boxing_data(dataset_npz_path, dataset_config_path, out_dir, frd_win_epochs, epochs=EPOCHS,
                  batchsize=batch_size)
