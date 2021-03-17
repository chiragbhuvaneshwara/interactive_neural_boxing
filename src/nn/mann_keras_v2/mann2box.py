import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import json

tf.keras.backend.set_floatx("float32")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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


def save_network(path, network, Xmean, Ymean, Xstd, Ystd):
    if os.path.exists(path):
        os.remove(path)

    if not os.path.exists(path):
        os.makedirs(path)

    print("save to : ", path)
    network.save(os.path.join(path, "model"))
    tf.saved_model.save(network, os.path.join(path, "saved_model"))
    if not os.path.exists(os.path.join(path, "means")):
        os.makedirs(os.path.join(path, "means"))
    Xmean.astype("float32").tofile(os.path.join(path, "means", "Xmean.bin"))
    Ymean.astype("float32").tofile(os.path.join(path, "means", "Ymean.bin"))
    Xstd.astype("float32").tofile(os.path.join(path, "means", "Xstd.bin"))
    Ystd.astype("float32").tofile(os.path.join(path, "means", "Ystd.bin"))


def LOAD_MANN(path):
    mann2 = tf.keras.models.load_model(path.replace(".h5", ""), custom_objects={"MANN": MANN}, compile=False)
    mann2.batch_size = 1
    print("model loaded")
    return mann2


def getVariationGating(network, input, batch_size=32):
    gws = []
    for i in range(10):
        bi = input[i * 32:(i + 1) * 32, :]
        out = network(bi)
        gws.append(out[:, -6:])
    #
    # gws.append(tensor.numpy())
    print("\nChecking the gating variability: ")
    print("  mean: ", np.mean(np.concatenate(gws, axis=0), axis=0))
    print("  std: ", np.std(np.concatenate(gws, axis=0), axis=0))
    print("  max: ", np.max(np.concatenate(gws, axis=0), axis=0))
    print("  min: ", np.min(np.concatenate(gws, axis=0), axis=0))
    print("")


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


def TrainAI4AnimationData(data_npz_path, data_config_path, output_folder, checkpoint_folder, epochs=30, batchsize=32):
    norm_path = os.path.join(output_folder, 'data_norm.json')

    with open(data_config_path) as f:
        dataset_config = json.load(f)

    bone_map = dataset_config['bone_map']
    col_demarcation_ids = dataset_config['col_demarcation_ids']
    x_col_demarcation = col_demarcation_ids[0]

    gating_indices = get_gating_indices(x_col_demarcation, bone_map)

    X, Y, norm = prepare_mann_data(data_npz_path, dataset_config)
    Xmean = np.array(norm['x_mean'], dtype=np.float64)
    Xstd = np.array(norm['x_std'], dtype=np.float64)
    Ymean = np.array(norm['y_mean'], dtype=np.float64)
    Ystd = np.array(norm['y_std'], dtype=np.float64)

    learning_rate = tf.keras.experimental.CosineDecayRestarts(0.0001, 10 * (len(X) // batchsize))
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    logdir = os.path.join(output_folder, "logs")
    epoch_dir = os.path.join(output_folder, "epochs")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    epoch_dir = os.path.join(epoch_dir, "epoch_%d")

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
        def __init__(self, X):
            super().__init__()
            self.X = X

        def on_epoch_begin(self, epoch, logs=None):
            # print("epoch start")
            # a = 0
            getVariationGating(self.model, self.X)

    cp_callback = EpochWriter(epoch_dir, checkpoint_folder, Xmean, Ymean, Xstd, Ystd)

    n_controls = 6
    epochs_executed = 0

    input_dim = X.shape[1]
    # gating_input_dim = X.shape[1] - cutindid
    output_dim = Y.shape[1]

    network = MANN(input_dim, output_dim, 512, 64, 6, gating_indices)

    # network(np.random.random((32, 498)))
    # network.summary()
    # print("conf: ", network.get_config())

    def loss_func(y, yt):
        yt = yt[:, :-6]
        gt = yt[:, -6:]
        rec_loss = tf.reduce_mean((y - yt) ** 2)

        # gt_loss = 1 / tf.exp(tf.reduce_mean(tf.math.reduce_std(gt, axis=0) ** 2))

        return rec_loss  # + 0.1 * gt_loss

    network.compile(optimizer=optimizer, loss=loss_func)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir), write_graph=True, write_images=False,
                                                 histogram_freq=0, update_freq="batch")

    # X = X[:(len(X) // batchsize) * batchsize, :]
    X = X[:(500 // batchsize) * batchsize, :]
    Y = Y[:len(X), :]

    print(X.shape, input_dim, output_dim)

    print("start training: ", X.shape, Y.shape)
    # network(X[:batchsize,:])

    gating_checker = GatingChecker(X)

    network.fit(X, Y, epochs=epochs, batch_size=batchsize, callbacks=[tensorboard, cp_callback, gating_checker],
                initial_epoch=epochs_executed)


if __name__ == '__main__':
    current_timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    # TrainNetwork("data/processed/pfnn.npz", os.path.join("training", current_timestamp, "out"),os.path.join("training", current_timestamp, "check"))
    TrainAI4AnimationData("data/boxing_fr_1_15/train.npz",
                          "data/boxing_fr_1_15/config.json",
                          os.path.join("training_nsm", current_timestamp, "out"), \
                          os.path.join("training", current_timestamp, "check"), \
                          epochs=2)
