import numpy as np
import tensorflow as tf
import os

from train.nn.mann_keras.mann import MANN


def prepare_mann_data(dataset, dataset_config):
    data = np.load(dataset)
    x_train = data["x"]
    y_train = data["y"]

    x_col_demarcation_ids = dataset_config['col_demarcation_ids'][0]
    y_col_demarcation_ids = dataset_config['col_demarcation_ids'][1]

    x_mean = np.mean(x_train, axis=0)
    y_mean = np.mean(y_train, axis=0)
    x_std = np.std(x_train, axis=0)
    y_std = np.std(y_train, axis=0)

    for k, v in x_col_demarcation_ids.items():
        x_std[v[0]: v[1]] = x_std[v[0]: v[1]].mean()

    for k, v in y_col_demarcation_ids.items():
        y_std[v[0]: v[1]] = y_std[v[0]: v[1]].mean()

    x_std[x_std == 0] = 0.0001
    y_std[y_std == 0] = 0.0001

    norm = {"x_mean": x_mean.tolist(),
            "y_mean": y_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_std": y_std.tolist()}

    x_train_norm = (x_train - x_mean) / x_std
    y_train_norm = (y_train - y_mean) / y_std

    return x_train_norm, y_train_norm, norm


def save_network(path, network, x_in_mean, y_out_mean, x_in_std, y_out_std):
    if os.path.exists(path):
        os.remove(path)

    if not os.path.exists(path):
        os.makedirs(path)

    network.save(os.path.join(path, "model"))
    tf.saved_model.save(network, os.path.join(path, "saved_model"))
    if not os.path.exists(os.path.join(path, "means")):
        os.makedirs(os.path.join(path, "means"))
    x_in_mean.astype("float32").tofile(os.path.join(path, "means", "Xmean.bin"))
    y_out_mean.astype("float32").tofile(os.path.join(path, "means", "Ymean.bin"))
    x_in_std.astype("float32").tofile(os.path.join(path, "means", "Xstd.bin"))
    y_out_std.astype("float32").tofile(os.path.join(path, "means", "Ystd.bin"))


def load_mann(path):
    mann2 = tf.keras.models.load_model(path.replace(".h5", ""), custom_objects={"MANN": MANN}, compile=False)
    mann2.batch_size = 1
    print("model loaded")
    return mann2


def load_binary(path):
    return np.fromfile(path, dtype=np.float32)


def mse_loss_fixed_gating(y, yt):
    yt = yt[:, :-6]
    rec_loss = tf.reduce_mean((y - yt) ** 2)
    return rec_loss


def mse_loss_wrapper(num_expert_nodes):
    def mse_loss_variable_gating(y, yt):
        yt = yt[:, :-num_expert_nodes]
        rec_loss = tf.reduce_mean((y - yt) ** 2)
        return rec_loss

    return mse_loss_variable_gating
