import json

import numpy as np
import tensorflow as tf
import os


class MANN(tf.keras.Model):
    def __init__(self, input_size, output_size, hidden_size, gating_hidden, expert_nodes, gating_indices, batch_size=32,
                 dropout_prob=0.2):
        super(MANN, self).__init__(name="mann_network")
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.gating_hidden = gating_hidden
        self.dropout_prob = dropout_prob

        self.weight_knots = []
        self.bias_knots = []
        self.expert_nodes = expert_nodes
        self.expert_weights = []

        self.batch_size = batch_size
        self.gating_indices = gating_indices
        self.gating_input = len(gating_indices)

    def get_config(self):
        # config = super(MANN, self).get_config()
        config = {"input_size": self.input_size, "output_size": self.output_size, "hidden_size": self.hidden_size,
                  "gating_hidden": self.gating_hidden, "expert_nodes": self.expert_nodes,
                  "gating_indices": self.gating_indices, "batch_size": 1, "dropout_prob": self.dropout_prob}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def create_weight(self, shape, name):
        weight_bound = np.sqrt(6. / np.sum(shape[-2:]))
        self.weight_knots.append(self.add_weight(shape=shape,
                                                 initializer=tf.keras.initializers.RandomUniform(-weight_bound,
                                                                                                 weight_bound),
                                                 trainable=True, dtype="float32", name=name))

    def create_bias(self, shape, name):
        self.bias_knots.append(self.add_weight(shape=shape,
                                               initializer=tf.keras.initializers.Zeros(),
                                               trainable=True, dtype="float32", name=name))

    def build(self, input_shape):
        self.create_weight([self.gating_hidden, self.gating_input], "g1_w")
        self.create_weight([self.gating_hidden, self.gating_hidden], "g2_w")
        self.create_weight([self.expert_nodes, self.gating_hidden], "g3_w")

        self.create_bias([self.gating_hidden, 1], "g1_b")
        self.create_bias([self.gating_hidden, 1], "g2_b")
        self.create_bias([self.expert_nodes, 1], "g3_b")

        self.create_weight([self.expert_nodes, self.hidden_size, self.input_size], "g1_w")
        self.create_weight([self.expert_nodes, self.hidden_size, self.hidden_size], "g2_w")
        self.create_weight([self.expert_nodes, self.output_size, self.hidden_size], "g3_w")

        self.create_bias([self.expert_nodes, self.hidden_size, 1], "g1_b")
        self.create_bias([self.expert_nodes, self.hidden_size, 1], "g2_b")
        self.create_bias([self.expert_nodes, self.output_size, 1], "g3_b")

    @tf.function
    def interpolate(self, experts, expert_weights):
        with tf.name_scope("interpolate"):
            e = tf.expand_dims(experts, 0)
            e = tf.tile(e, [self.batch_size, 1, 1, 1])
            w = tf.expand_dims(tf.expand_dims(expert_weights, -1), -1)
            # print("interpolate: ", w, e)
            r = w * e
            return tf.reduce_sum(r, axis=1)

    @tf.function
    def gating_network(self, inputs, training=None):
        if training:
            print("training not none")
            H0 = tf.nn.dropout(inputs, self.dropout_prob)
        else:
            H0 = inputs

        # print("gh1 " , self.weight_knots[0], H0)
        # print(tf.matmul(self.weight_knots[0], H0), self.bias_knots[0])
        H1 = tf.matmul(self.weight_knots[0], H0) + self.bias_knots[0]
        H1 = tf.nn.elu(H1)
        if not training is None:
            H1 = tf.nn.dropout(H1, self.dropout_prob)

        # print("gh2 " , self.weight_knots[1],  H1)
        # print(tf.matmul(self.weight_knots[1], H1), self.bias_knots[1])
        H2 = tf.matmul(self.weight_knots[1], H1) + self.bias_knots[1]
        H2 = tf.nn.elu(H2)
        if not training is None:
            H2 = tf.nn.dropout(H2, self.dropout_prob)

        # print("gh3 " , self.weight_knots[2], H2)
        # print(tf.matmul(self.weight_knots[2], H2), self.bias_knots[2])
        H3 = tf.matmul(self.weight_knots[2], H2) + self.bias_knots[2]
        H3 = tf.nn.softmax(H3, axis=1)
        return H3

    @tf.function
    def motion_network(self, inputs, expert_weights, training=None):
        if not training is None:
            H0 = tf.nn.dropout(inputs, self.dropout_prob)
        else:
            H0 = inputs

        w0 = self.interpolate(self.weight_knots[3], expert_weights)
        b0 = self.interpolate(self.bias_knots[3], expert_weights)

        # print("H1 " , w0, H0)
        # print(tf.matmul(w0, H0), b0)

        H1 = tf.matmul(w0, H0) + b0
        H1 = tf.nn.elu(H1)
        if not training is None:
            H1 = tf.nn.dropout(H1, self.dropout_prob)

        w1 = self.interpolate(self.weight_knots[4], expert_weights)
        b1 = self.interpolate(self.bias_knots[4], expert_weights)

        # print("H2 " , w1, H1)
        # print(tf.matmul(w1, H1), b1)

        H2 = tf.matmul(w1, H1) + b1
        H2 = tf.nn.elu(H2)
        if not training is None:
            H2 = tf.nn.dropout(H2, self.dropout_prob)

        w2 = self.interpolate(self.weight_knots[5], expert_weights)
        b2 = self.interpolate(self.bias_knots[5], expert_weights)

        # print("H3 " , w2, H2)
        # print(tf.matmul(w2, H2), b2)

        H3 = tf.matmul(w2, H2) + b2

        return H3

    @tf.function
    def call(self, inputs, training=None):
        # self.batch_size.assign(inputs.shape[0])

        expert_input = tf.expand_dims(tf.gather(inputs, self.gating_indices, axis=1), -1)
        motion_input = tf.expand_dims(inputs, -1)
        # print("einpput:", expert_input)
        expert_weights = self.gating_network(expert_input, training)[..., 0]
        # print("gating: ", expert_weights)
        # print("motion in:", motion_input)
        output = self.motion_network(motion_input, expert_weights, training)[..., 0]

        return tf.concat([output, expert_weights], axis=-1)


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

    x_std[x_std == 0] = 0.01
    y_std[y_std == 0] = 0.01

    norm = {"x_mean": x_mean.tolist(),
            "y_mean": y_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_std": y_std.tolist()}

    # TODO Setup comparator to warn if input and inverse of output are the same or not

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


def get_variation_gating(network, input_data, batch_size):
    gws = []
    r_lim = (input_data.shape[0] - 1) // batch_size
    for i in range(r_lim):
        bi = input_data[i * batch_size:(i + 1) * batch_size, :]
        out = network(bi)
        # TODO Setup var for extracting gating outputs
        gws.append(out[:, -6:])
    #
    # gws.append(tensor.numpy())
    print("\nChecking the gating variability: ")
    print("  mean: ", np.mean(np.concatenate(gws, axis=0), axis=0))
    print("  std: ", np.std(np.concatenate(gws, axis=0), axis=0))
    print("  max: ", np.max(np.concatenate(gws, axis=0), axis=0))
    print("  min: ", np.min(np.concatenate(gws, axis=0), axis=0))
    print("")


class EpochWriter(tf.keras.callbacks.Callback):
    # def __init__(self, path, checkpoint_folder, Xmean, Ymean, Xstd, Ystd):
    def __init__(self, path, Xmean, Ymean, Xstd, Ystd):
        super().__init__()
        self.path = path
        # self.checkpoint_folder = os.path.join(checkpoint_folder, "epoch_%d")
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


def loss_func(y, yt):
    # TODO: What is happening here? Can probably redefine this function yourself
    yt = yt[:, :-6]
    rec_loss = tf.reduce_mean((y - yt) ** 2)
    return rec_loss
