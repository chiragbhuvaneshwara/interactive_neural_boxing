import json
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import shutil
import argparse

from train.nn.mann_keras.mann import MANN
from train.nn.mann_keras.callbacks import EpochWriter, GatingChecker
from train.nn.mann_keras.utils import prepare_mann_data, mse_loss_wrapper

tf.keras.backend.set_floatx("float32")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_gating_indices(x_ids, joint_ids):
    """
    The gating input is a subset of the motion network input.
    This function returns the exact index positions of the subset to be extracted from the motion network input.

    @param x_ids: column demarcation ids for every var present in motion network input i.e. local_pos: [start_idx, end_idx]
    @param joint_ids: dict of joint name and idx in bvh skeleton hierarchy
    @return: gating_ids: list of index positions from where gating input is to be extracted from motion network input
    """

    def _generate_id_sequence(column_demarcation_ids, key):
        return [i for i in range(column_demarcation_ids[key][0], column_demarcation_ids[key][1])]

    wrist_velocities_tr_ids = _generate_id_sequence(x_ids, 'x_right_wrist_vels_tr') + \
                              _generate_id_sequence(x_ids, 'x_left_wrist_vels_tr')

    # punch_labels_tr_ids = _generate_id_sequence(x_ids, 'x_right_punch_labels_tr') + \
    #                       _generate_id_sequence(x_ids, 'x_left_punch_labels_tr')

    current_punch_labels_ids = _generate_id_sequence(x_ids, 'x_right_punch_labels') + \
                               _generate_id_sequence(x_ids, 'x_left_punch_labels')

    punch_target_ids = _generate_id_sequence(x_ids, 'x_right_punch_target') + \
                       _generate_id_sequence(x_ids, 'x_left_punch_target')

    f_r_id = joint_ids["RightAnkle"] * 3
    f_l_id = joint_ids["LeftAnkle"] * 3
    foot_end_effector_velocities_ids = _generate_id_sequence(x_ids, 'x_local_vel')[f_r_id: f_r_id + 3] + \
                                       _generate_id_sequence(x_ids, 'x_local_vel')[f_l_id: f_l_id + 3]
    w_r_id = joint_ids["RightWrist"] * 3
    w_l_id = joint_ids["LeftWrist"] * 3
    wrist_end_effector_velocities_ids = _generate_id_sequence(x_ids, 'x_local_vel')[w_r_id: w_r_id + 3] + \
                                        _generate_id_sequence(x_ids, 'x_local_vel')[w_l_id: w_l_id + 3]

    # Gating input for OG MANN:
    # foot end effector velocities,
    # the current action variables
    # and the desired velocity of the character
    # TODO: try not to include punch targets
    # gating_ids = wrist_velocities_tr_ids + \
    #              punch_labels_tr_ids + \
    #              punch_target_ids + \
    #              wrist_end_effector_velocities_ids \
    #              + \
    #              current_punch_labels_ids + \
    #              foot_end_effector_velocities_ids
    gating_ids = wrist_velocities_tr_ids + \
                 punch_target_ids + \
                 wrist_end_effector_velocities_ids \
                 + \
                 current_punch_labels_ids + \
                 foot_end_effector_velocities_ids
    # desired_vel (part of OG MANN gating input)
    return gating_ids


def train_boxing_data(data_npz_path, data_config_path, output_dir, frd_win_epochs_data, epochs=30, batchsize=32):
    """
    Trains a MANN keras model on supplied boxing data which is processed by neural_data_prep module and is now in
    a format that can be fed into a neural network.

    @param data_npz_path: str, path to boxing data in neural network supported format
    @param data_config_path: str, path to dict containing parameters used to generate dataset and some info for
    accessing different variables in input and output vectors.
    @param output_dir: str, path to save training logs and models
    @param frd_win_epochs_data: str, containing frame rate div, trajectory window and num epochs parameters
    @param epochs: num epochs to train the model
    @param batchsize: batch size of data to be used for one gradient update
    """
    logdir = os.path.join(output_dir, "logs")
    epoch_dir = os.path.join(output_dir, "epochs")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, frd_win_epochs_data + '.json'), 'w') as outfile:
        json.dump(frd_win_epochs_data, outfile, indent=4)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    epoch_dir = os.path.join(epoch_dir, "epoch_%d")

    with open(data_config_path) as f:
        dataset_config = json.load(f)

    bone_map = dataset_config['bone_map']
    col_demarcation_ids = dataset_config['col_demarcation_ids']
    x_col_demarcation = col_demarcation_ids[0]

    gating_indices = get_gating_indices(x_col_demarcation, bone_map)

    X, Y, norm = prepare_mann_data(data_npz_path, dataset_config)
    x_mean = np.array(norm['x_mean'], dtype=np.float64)
    x_std = np.array(norm['x_std'], dtype=np.float64)
    y_mean = np.array(norm['y_mean'], dtype=np.float64)
    y_std = np.array(norm['y_std'], dtype=np.float64)

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    learning_rate = tf.keras.experimental.CosineDecayRestarts(0.0001, 10 * (len(X) // batchsize))
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    num_expert_nodes = 6
    network = MANN(input_dim, output_dim, 512, 64, num_expert_nodes, gating_indices, batch_size=batchsize)
    network.compile(optimizer=optimizer, loss=mse_loss_wrapper(num_expert_nodes))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir), write_graph=True, write_images=False,
                                                 histogram_freq=0, update_freq="batch")
    cp_callback = EpochWriter(epoch_dir, x_mean, y_mean, x_std, y_std)
    X = X[:(len(X) // batchsize) * batchsize, :]
    # X = X[:(100 // batchsize) * batchsize, :]
    Y = Y[:len(X), :]
    gating_checker = GatingChecker(X, batchsize)
    epochs_executed = 0

    print(X.shape, input_dim, output_dim)
    print("start training: ", X.shape, Y.shape)
    network.fit(X, Y, epochs=epochs, batch_size=batchsize, callbacks=[tensorboard, cp_callback, gating_checker],
                initial_epoch=epochs_executed)
    print("Training completed. Models stored in: \n", output_dir)

# if __name__ == '__main__':
#     args_parser = argparse.ArgumentParser()
#     args_parser.add_argument("-d", "--develop", help="Run on subset",
#                              action="store_true", default=False)
#     args_parser.add_argument("-l", "--local", help="Flag indicating remote machine or local machine",
#                              action="store_true", default=False)
#     args = args_parser.parse_args()
#     DEVELOP = args.develop
#     LOCAL = args.local
#
#     EPOCHS = 100
#     FRD = 1
#     WINDOW = 15
#     OUT_BASE_PATH = os.path.join("train", "models", "mann_tf2_v2")
#     ############################################
#     frd_win = 'boxing_fr_' + str(FRD) + '_' + str(WINDOW)
#     current_timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
#
#     if not DEVELOP:
#         dataset_config_path = os.path.join("data", "neural_data", frd_win, "config.json")
#         dataset_npz_path = os.path.join("data", "neural_data", frd_win, "train.npz")
#         batch_size = 32
#
#     elif DEVELOP:
#         print('Dev Mode')
#         OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "dev")
#         dataset_config_path = os.path.join("data", "neural_data", "dev", frd_win, "config.json")
#         dataset_npz_path = os.path.join("data", "neural_data", "dev", frd_win, "train.npz")
#         batch_size = 2
#         EPOCHS = 2
#
#     frd_win_epochs = frd_win + '_' + str(EPOCHS)
#     if LOCAL:
#         print('Local machine dev')
#         OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "local_dev_saved_models")
#         if os.path.exists(OUT_BASE_PATH):
#             shutil.rmtree(OUT_BASE_PATH, ignore_errors=False, onerror=None)
#             os.mkdir(OUT_BASE_PATH)
#         dataset_config_path = os.path.join("data", "neural_data", "dev", frd_win, "config.json")
#         dataset_npz_path = os.path.join("data", "neural_data", "dev", frd_win, "train.npz")
#         batch_size = 2
#         EPOCHS = 2
#         out_dir = os.path.join(OUT_BASE_PATH)
#
#     elif not LOCAL:
#         out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp)
#
#     train_boxing_data(dataset_npz_path, dataset_config_path, out_dir, frd_win_epochs, epochs=EPOCHS,
#                       batchsize=batch_size)
