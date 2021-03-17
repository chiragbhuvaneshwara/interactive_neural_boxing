import json
import os
import glob
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from src.nn.mann_keras_v2.mann import MANN, loss_func, prepare_mann_data, get_variation_gating, save_network
    # EpochWriter, GatingChecker

tf.keras.backend.set_floatx("float32")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def setup_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    print('Created:', path)


def get_gating_indices(x_ids, joint_ids):
    def _generate_id_sequence(column_demarcation_ids, key):
        return [i for i in range(column_demarcation_ids[key][0], column_demarcation_ids[key][1])]

    wrist_velocities_tr_ids = _generate_id_sequence(x_ids, 'x_right_wrist_vels_tr') + \
                              _generate_id_sequence(x_ids, 'x_left_wrist_vels_tr')

    punch_labels_tr_ids = _generate_id_sequence(x_ids, 'x_right_punch_labels_tr') + \
                          _generate_id_sequence(x_ids, 'x_left_punch_labels_tr')

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
    gating_ids = wrist_velocities_tr_ids + \
                 punch_labels_tr_ids + \
                 punch_target_ids + \
                 wrist_end_effector_velocities_ids \
                 + \
                 current_punch_labels_ids + \
                 foot_end_effector_velocities_ids
    # desired_vel (part of OG MANN gating input)
    return gating_ids


def train_boxing_data(data_npz_path, data_config_path, output_dir, checkpoint_dir, epochs=30, batchsize=32):
    # logdir = setup_dir(os.path.join(output_dir, "logs"))
    # epoch_dir = setup_dir(os.path.join(output_dir, "epochs"))
    # epoch_dir = os.path.join(epoch_dir, "epoch_%d")
    # checkpoint_dir = setup_dir(os.path.join(output_dir, "checkpoints"))
    logdir = os.path.join(output_dir, "logs")
    epoch_dir = os.path.join(output_dir, "epochs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    epoch_dir = os.path.join(epoch_dir, "epoch_%d")

    norm_path = os.path.join(output_dir, 'data_norm.json')

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

    network = MANN(input_dim, output_dim, 512, 64, 6, gating_indices, batch_size=batchsize)
    network.compile(optimizer=optimizer, loss=loss_func)

    class EpochWriter(tf.keras.callbacks.Callback):
        def __init__(self, path, checkpoint_folder, Xmean, Ymean, Xstd, Ystd):
            super().__init__()
            self.path = path
            self.checkpoint_folder = os.path.join(checkpoint_folder, "epoch_%d")
            self.Xmean = Xmean
            self.Ymean = Ymean
            self.Xstd = Xstd
            self.Ystd = Ystd

        def on_epoch_end(self, epoch, logs=None):
            save_network(self.path % epoch, self.model, self.Xmean, self.Ymean, self.Xstd, self.Ystd)
            print("\nModel saved to ", self.path % epoch)

    class GatingChecker(tf.keras.callbacks.Callback):
        def __init__(self, X, batch_size):
            super().__init__()
            self.X = X
            self.batch_size = batch_size

        def on_epoch_begin(self, epoch, logs=None):
            # print("epoch start")
            # a = 0
            get_variation_gating(self.model, self.X, self.batch_size)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir), write_graph=True, write_images=False,
                                                 histogram_freq=0, update_freq="batch")
    cp_callback = EpochWriter(epoch_dir, checkpoint_dir, x_mean, y_mean, x_std, y_std)
    # X = X[:(len(X) // batchsize) * batchsize, :]
    X = X[:(500 // batchsize) * batchsize, :]
    Y = Y[:len(X), :]
    gating_checker = GatingChecker(X, batchsize)
    epochs_executed = 0

    print(X.shape, input_dim, output_dim)
    print("start training: ", X.shape, Y.shape)

    network.fit(X, Y, epochs=epochs, batch_size=batchsize, callbacks=[tensorboard, cp_callback, gating_checker],
                initial_epoch=epochs_executed)


if __name__ == '__main__':
    DEVELOP = False
    EPOCHS = 100
    FRD = 1
    WINDOW = 15
    OUT_BASE_PATH = os.path.join("models", "mann_tf2_v2")
    ############################################
    frd_win = 'boxing_fr_' + str(FRD) + '_' + str(WINDOW)
    current_timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    if not DEVELOP:
        # TODO Rename dataset config file in file system to dataset_config.json instead of config.json
        dataset_config_path = os.path.join("data", frd_win, "config.json")
        dataset_npz_path = os.path.join('data', frd_win, 'train.npz')
        batch_size = 32
        EPOCHS = 2

    elif DEVELOP:
        OUT_BASE_PATH = os.path.join(OUT_BASE_PATH, "dev")
        dataset_config_path = os.path.join("data", "dev", frd_win, "config.json")
        dataset_npz_path = os.path.join('data', 'dev', frd_win, 'train.npz')
        batch_size = 2
        EPOCHS = 2

    frd_win_epochs = frd_win + '_' + str(EPOCHS)
    # out_dir = setup_dir(os.path.join(OUT_BASE_PATH, frd_win_epochs))
    # out_dir = setup_dir(os.path.join(out_dir, current_timestamp))
    out_dir = setup_dir(os.path.join(OUT_BASE_PATH, frd_win_epochs, current_timestamp))
    checkpoint_dir = setup_dir(os.path.join("training", current_timestamp, "check"))
    # train_boxing_data(dataset_npz_path, dataset_config_path, out_dir, epoch_out_dir, epochs=EPOCHS, batchsize= batch_size)
    # train_boxing_data(dataset_npz_path, dataset_config_path, out_dir, checkpoint_dir, epochs=EPOCHS, batchsize=batch_size)
    train_boxing_data(dataset_npz_path, dataset_config_path,
                      os.path.join("training_nsm", current_timestamp, "out"),
                      os.path.join("training", current_timestamp, "check"),
                      epochs=EPOCHS, batchsize=batch_size)
