import json
import numpy as np
import tensorflow as tf
import os

from common.utils import retrieve_name

from train.nn.mann_keras.mann import MANN
from train.nn.mann_keras.callbacks import EpochWriter, GatingChecker
from train.nn.mann_keras.utils import prepare_mann_data, mse_loss_variable_gating

tf.keras.backend.set_floatx("float32")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_gating_indices(x_ids, joint_ids, traj_window_root, traj_window_wrist):
    """
    The gating input is a subset of the motion network input.
    This function returns the exact index positions of the subset to be extracted from the motion network input.

    @param x_ids: column demarcation ids for every var present in motion network input i.e. local_pos: [start_idx, end_idx]
    @param joint_ids: dict of joint name and idx in bvh skeleton hierarchy
    @return: gating_ids: list of index positions from where gating input is to be extracted from motion network input
    """

    def _generate_id_sequence(column_demarcation_ids, key):
        return [i for i in range(column_demarcation_ids[key][0], column_demarcation_ids[key][1])]

    def _get_past_curr_future_ids(sequence, multiplier, tr_window):
        sequence = sequence[:multiplier] + sequence[tr_window * multiplier:(tr_window + 1) * multiplier] + sequence[
                                                                                                           -multiplier:]
        return sequence

    right_wrist_velocities_tr_ids = _generate_id_sequence(x_ids, 'x_right_wrist_vels_tr')
    left_wrist_velocities_tr_ids = _generate_id_sequence(x_ids, 'x_left_wrist_vels_tr')
    right_wrist_velocities_tr_ids = _get_past_curr_future_ids(right_wrist_velocities_tr_ids, 3, traj_window_wrist)
    left_wrist_velocities_tr_ids = _get_past_curr_future_ids(left_wrist_velocities_tr_ids, 3, traj_window_wrist)
    wrist_velocities_tr_ids = right_wrist_velocities_tr_ids + left_wrist_velocities_tr_ids

    root_velocities_tr_ids = _generate_id_sequence(x_ids, 'x_root_vels_tr')
    root_velocities_tr_ids = _get_past_curr_future_ids(root_velocities_tr_ids, 2, traj_window_root)

    root_dirs_tr_ids = _generate_id_sequence(x_ids, 'x_root_dirs_tr')
    root_dirs_tr_ids = _get_past_curr_future_ids(root_dirs_tr_ids, 2, traj_window_root)

    root_pos_tr_ids = _generate_id_sequence(x_ids, 'x_root_pos_tr')
    root_pos_tr_ids = _get_past_curr_future_ids(root_pos_tr_ids, 2, traj_window_root)

    # punch_labels_tr_ids = _generate_id_sequence(x_ids, 'x_right_punch_labels_tr') + \
    #                       _generate_id_sequence(x_ids, 'x_left_punch_labels_tr')

    current_punch_labels_ids = _generate_id_sequence(x_ids, 'x_right_punch_labels') + \
                               _generate_id_sequence(x_ids, 'x_left_punch_labels')

    punch_target_ids = _generate_id_sequence(x_ids, 'x_right_punch_target') + \
                       _generate_id_sequence(x_ids, 'x_left_punch_target')

    f_r_a_id = joint_ids["RightAnkle"] * 3
    f_l_a_id = joint_ids["LeftAnkle"] * 3
    foot_ankle_velocities_ids = _generate_id_sequence(x_ids, 'x_local_vel')[f_r_a_id: f_r_a_id + 3] + \
                                _generate_id_sequence(x_ids, 'x_local_vel')[f_l_a_id: f_l_a_id + 3]
    foot_ankle_pos_ids = _generate_id_sequence(x_ids, 'x_local_pos')[f_r_a_id: f_r_a_id + 3] + \
                         _generate_id_sequence(x_ids, 'x_local_pos')[f_l_a_id: f_l_a_id + 3]
    f_r_t_id = joint_ids["RightToe"] * 3
    f_l_t_id = joint_ids["LeftToe"] * 3
    foot_toes_velocities_ids = _generate_id_sequence(x_ids, 'x_local_vel')[f_r_t_id: f_r_t_id + 3] + \
                               _generate_id_sequence(x_ids, 'x_local_vel')[f_l_t_id: f_l_t_id + 3]
    foot_toes_pos_ids = _generate_id_sequence(x_ids, 'x_local_pos')[f_r_t_id: f_r_t_id + 3] + \
                        _generate_id_sequence(x_ids, 'x_local_pos')[f_l_t_id: f_l_t_id + 3]
    w_r_id = joint_ids["RightWrist"] * 3
    w_l_id = joint_ids["LeftWrist"] * 3
    wrist_end_effector_velocities_ids = _generate_id_sequence(x_ids, 'x_local_vel')[w_r_id: w_r_id + 3] + \
                                        _generate_id_sequence(x_ids, 'x_local_vel')[w_l_id: w_l_id + 3]

    # Gating input for OG MANN:
    # foot end effector velocities,
    # the current action variables
    # and the desired velocity of the character
    # TODO: try not to include punch targets
    # TODO: increase traj window for walking
    gating_ids = [
        root_pos_tr_ids,
        root_dirs_tr_ids,
        wrist_velocities_tr_ids,
        # punch_target_ids,
        wrist_end_effector_velocities_ids,
        current_punch_labels_ids,
        foot_ankle_velocities_ids,
        foot_toes_velocities_ids,
        foot_ankle_pos_ids,
        foot_toes_pos_ids,
    ]
    # desired_vel (part of OG MANN gating input)

    gating_variables = list(map(retrieve_name, gating_ids))

    # Flattening gating ids list
    gating_ids = [item for sublist in gating_ids for item in sublist]

    return gating_ids, gating_variables


def train_boxing_data(data_config_path, output_dir, learning_rate=0.001, num_hidden_neurons=512, num_expert_nodes=6,
                      epochs=30, batchsize=32):
    """
    Trains a MANN keras model on supplied boxing data which is processed by neural_data_prep module and is now in
    a format that can be fed into a neural network.

    @param num_expert_nodes:
    @type num_expert_nodes:
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
    # with open(os.path.join(output_dir, frd_win_epochs_data + '.json'), 'w') as outfile:
    #     json.dump(frd_win_epochs_data, outfile, indent=4)

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
    tr_win_root = dataset_config["traj_window_root"]
    tr_win_wr = dataset_config["traj_window_wrist"]

    gating_indices, gating_variable_names = get_gating_indices(x_col_demarcation, bone_map, tr_win_root, tr_win_wr)

    X, Y, norm = prepare_mann_data(dataset_config)
    x_mean = np.array(norm['x_mean'], dtype=np.float64)
    x_std = np.array(norm['x_std'], dtype=np.float64)
    y_mean = np.array(norm['y_mean'], dtype=np.float64)
    y_std = np.array(norm['y_std'], dtype=np.float64)

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    # TODO experiment with network params to make loss curve steeper
    learning_rate = tf.keras.experimental.CosineDecayRestarts(learning_rate, 10 * (len(X) // batchsize))
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    training_details = {
        "learning_rate": {
            "name": type(learning_rate).__name__,
            "initial_rate": learning_rate.initial_learning_rate,
            "first_decay_steps": learning_rate.first_decay_steps
        },
        "num_experts": num_expert_nodes,
        "num_hidden_neurons": num_hidden_neurons,
        "gating_variables": gating_variable_names,
        "num_data_pts": X.shape[0],
        "optimizer": type(optimizer).__name__,
        "loss": mse_loss_variable_gating.__name__,
        "dataset_config": dataset_config,
        "dataset_config_path": data_config_path,
    }

    # TODO Try models with fewer hidden neurons Ex 256
    network = MANN(input_dim, output_dim, num_hidden_neurons, 64, num_expert_nodes, gating_indices,
                   batch_size=batchsize)
    network.compile(optimizer=optimizer, loss=mse_loss_variable_gating(num_expert_nodes))
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

    training_details.update(network.get_summary())
    with open(os.path.join(output_dir, "network_config.json"), "w") as f:
        json.dump(training_details, f, indent=4)

    print("Training completed. Models stored in: \n", output_dir)
    print(learning_rate.first_decay_steps)
