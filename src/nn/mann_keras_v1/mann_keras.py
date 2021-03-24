import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

from src.nn.mann_keras_v1.network_instantiater import TrainNetwork, InstantiateNetworkFromConfig
from sklearn import preprocessing

from zipfile import ZipFile
# TODO Replace with mann_2
# TODO Verify and understand if MANN network is setup correctly
# TODO Verify if required ==> Doesn't this mean that the weights are always initialized the same way?
rng = np.random.RandomState(23456)


class MotionPredictionLayer(tf.keras.layers.Layer):
    def __init__(self, shape, rng, dropout_rate, activation, name=None, use_discrete_weights=False):
        super(MotionPredictionLayer, self).__init__(name=name)
        self.n_knots = shape[0]
        self.weight_shape = shape
        self.bias_shape = shape[:-1]
        self.rng = rng
        self.dropout = dropout_rate
        self.activation = activation
        self.weight_knots = tf.Variable(self.initial_weight_knots, name="weight_knots")
        self.bias_knots = tf.Variable(np.zeros(self.bias_shape, dtype=np.float32), name='bias_knots')
        self.use_discrete_weights = use_discrete_weights

    def initial_weight_knots(self):
        bound = np.sqrt(6. / np.prod(self.weight_shape[-2:]))
        return np.asarray(self.rng.uniform(low=-bound, high=bound, size=self.weight_shape), dtype=np.float32)

    def call(self, inputs, training=False):
        # inputs = np.array(number_of_batches, input_dimension)
        # inputs[:, 0:n_input]  - input
        # inputs[:, n_input:] - gating weights
        p = tf.identity(inputs[:, -(self.n_knots):])
        # phase = inputs[0] # blending coefficients
        # weights knots  shape = (4, 512) --> afterwards wi shape (512)
        # wi = tf.cast(phase[:,0] * 0, tf.float32) + 1 * self.weight_knots[0]
        # bi = tf.cast(phase[:,0] * 0, tf.float32) + 1 * self.bias_knots[0]

        with tf.name_scope("interpolation") as scope:
            zeros = tf.cast(p * 0, tf.int32)[:, 0]
            phase = tf.expand_dims(tf.expand_dims(p, 1), 1)
            ones = zeros + 1
            wi = tf.math.multiply(phase[:, :, :, 0], tf.nn.embedding_lookup(self.weight_knots, zeros))
            bi = tf.math.multiply(phase[:, 0, :, 0], tf.nn.embedding_lookup(self.bias_knots, zeros))

            # wi = phase[:,0,tf.newaxis] * self.weight_knots[0]
            for i in range(1, self.n_knots):
                wi = wi + tf.math.multiply(phase[:, :, :, i], tf.nn.embedding_lookup(self.weight_knots, ones * i))
                # + phase[:,1,tf.newaxis] * self.weight_knots[1] + phase[:,2,tf.newaxis] * self.weight_knots[2] + phase[:,3,tf.newaxis] * self.weight_knots[3]#tf.matmul(phase, w)
                bi = bi + tf.math.multiply(phase[:, 0, :, i], tf.nn.embedding_lookup(self.bias_knots, ones * i))
            # multiplication of the layer
            bi = tf.expand_dims(bi, -1)
        with tf.name_scope("multiplication") as scope:
            res = tf.matmul(wi, inputs[:, :-self.n_knots, tf.newaxis]) + bi
            if self.activation is not None:
                res = self.activation(res)
            if training:
                res = layers.Dropout(self.dropout)(res)
        res = tf.squeeze(res, -1)
        phase = tf.squeeze(tf.squeeze(phase, 1), 1)
        return res

    def export_binary_weights(self, save_path, prefix='', suffix=''):
        with ZipFile(save_path, "a") as zipf:
            with zipf.open('weights/{}W{}.bin'.format(prefix, suffix), "w") as f:
                f.write(np.array(self.weight_knots.numpy(), dtype=np.float32).tobytes())
            with zipf.open('weights/{}b{}.bin'.format(prefix, suffix), "w") as f:
                f.write(np.array(self.bias_knots.numpy(), dtype=np.float32).tobytes())

    def load_binary_weights(self, path, prefix='', suffix=''):
        """
        Loads a network from a zip file (path)

        Arguments:
            path {[type_in]} -- path to zip file

        Keyword Arguments:
            prefix {str} -- [description] (default: {''})
            suffix {str} -- [description] (default: {''})
        """

        with ZipFile(path) as zip:
            with zip.open("weights/{}W{}.bin".format(prefix, suffix), "r") as w:
                weight_value = np.frombuffer(w.read(), dtype=np.float32)
            with zip.open("weights/{}b{}.bin".format(prefix, suffix), "r") as w:
                bias_value = np.frombuffer(w.read(), dtype=np.float32)

        # weight_value = np.fromfile(os.path.join(path, '{}W{}.bin'.format(prefix, suffix)), np.float32)
        # bias_value = np.fromfile(os.path.join(path, '{}b{}.bin'.format(prefix, suffix)), np.float32)
        self.weight_knots.assign(np.reshape(weight_value, self.weight_shape))
        self.bias_knots.assign(np.reshape(bias_value, self.bias_shape))


class MANN(tf.keras.Model):
    def __init__(self, config):
        name = "mann" if not "name" in config else config["name"]
        n_controls = config["n_controls"]
        self.input_dim = config["input_dim"]
        input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        output_dim = config["output_dim"]
        dropout_rate = config["dropout_rate"]
        hidden_dim = config["hidden_dim"]
        gating_indices = config["gating_indices"]
        gating_hidden_dim = config["gating_hidden_dim"]

        super(MANN, self).__init__(name=name)
        self.norm = config["norm"]
        self.n_controls = n_controls
        self.hidden_dim = hidden_dim
        self.gating_indices = list(gating_indices)
        self.gating_hidden_dim = gating_hidden_dim
        self.gating_output_dim = n_controls

        self.gating_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.glayer1 = tf.keras.layers.Dense(self.gating_hidden_dim, activation='elu',
                                             input_dim=len(self.gating_indices), name="GL0")
        self.glayer2 = tf.keras.layers.Dense(self.gating_hidden_dim, activation='elu', input_dim=self.gating_hidden_dim,
                                             name="GL1")
        self.glayer3 = tf.keras.layers.Dense(self.gating_output_dim, activation='linear',
                                             input_dim=self.gating_hidden_dim, name="GL2")

        self.layer1 = MotionPredictionLayer(shape=(n_controls, self.hidden_dim, input_dim), rng=rng,
                                            dropout_rate=dropout_rate, activation=tf.nn.elu, name='ML0')
        self.layer2 = MotionPredictionLayer(shape=(n_controls, self.hidden_dim, self.hidden_dim), rng=rng,
                                            dropout_rate=dropout_rate, activation=tf.nn.elu, name='ML1')
        self.layer3 = MotionPredictionLayer(shape=(n_controls, output_dim, self.hidden_dim), rng=rng,
                                            dropout_rate=dropout_rate, activation=None, name='ML2')

    def call(self, inputs, training=False):
        # scope 1 - input prepare
        with tf.name_scope("gatingnetwork") as scope:
            gating_input = tf.gather(inputs, self.gating_indices, axis=-1)
            # gating_input = tf.expand_dims(gating_input, -1)
            # scope 2 - gating network
            glayer1_output = self.glayer1(gating_input)
            if (training):
                glayer1_output = self.gating_dropout(glayer1_output)
            glayer2_output = self.glayer2(glayer1_output)
            if (training):
                glayer2_output = self.gating_dropout(glayer2_output)
            gating_weights = self.glayer3(glayer2_output)
            if (training):
                gating_weights = self.gating_dropout(gating_weights)

        with tf.name_scope("motionnetwork") as scope:
            # scope 3 - motion network
            # dont forget to add the phase to each layer
            layer1_output = self.layer1(tf.concat([inputs, gating_weights], axis=-1), training)
            layer2_output = self.layer2(tf.concat([layer1_output, gating_weights], axis=-1), training)
            output = self.layer3(tf.concat([layer2_output, gating_weights], axis=-1), training)
        return output

    def export_gating_weight(self, layer, save_path, prefix='', suffix=''):
        with ZipFile(save_path, "a") as zipf:
            with zipf.open('weights/{}GW{}.bin'.format(prefix, suffix), "w") as f:
                f.write(np.array(layer.trainable_variables[0].numpy(), dtype=np.float32).tobytes())
            with zipf.open('weights/{}Gb{}.bin'.format(prefix, suffix), "w") as f:
                f.write(np.array(layer.trainable_variables[1].numpy(), dtype=np.float32).tobytes())

    def load_gating_weight(self, layer, shape, path, prefix='', suffix=''):
        """
        Loads a network from a zip file (path)

        Arguments:
            path {[type_in]} -- path to zip file

        Keyword Arguments:
            prefix {str} -- [description] (default: {''})
            suffix {str} -- [description] (default: {''})
        """

        with ZipFile(path) as zip:
            with zip.open("weights/{}GW{}.bin".format(prefix, suffix), "r") as w:
                weight_value = np.frombuffer(w.read(), dtype=np.float32)
            with zip.open("weights/{}Gb{}.bin".format(prefix, suffix), "r") as w:
                bias_value = np.frombuffer(w.read(), dtype=np.float32)

        # weight_value = np.fromfile(os.path.join(path, '{}W{}.bin'.format(prefix, suffix)), np.float32)
        # bias_value = np.fromfile(os.path.join(path, '{}b{}.bin'.format(prefix, suffix)), np.float32)
        layer.build(np.flip(shape))
        layer.variables[0].assign(np.reshape(weight_value, shape))
        layer.variables[1].assign(np.reshape(bias_value, (shape[-1])))

    def export_binary_weights(self, save_path):
        """save control knots: W0, b0, W1, b1, W2, b2

        """
        self.export_gating_weight(self.glayer1, save_path, suffix="0")
        self.export_gating_weight(self.glayer2, save_path, suffix="1")
        self.export_gating_weight(self.glayer3, save_path, suffix="2")

        self.layer1.export_binary_weights(save_path, suffix='0')
        self.layer2.export_binary_weights(save_path, suffix='1')
        self.layer3.export_binary_weights(save_path, suffix='2')

    def load_binary_weights(self, path):
        # self.load_gating_weight(self.glayer1, (self.gating_hidden_dim, len(self.gating_indices)), path, suffix="0")
        # self.load_gating_weight(self.glayer2, (self.gating_hidden_dim, self.gating_hidden_dim), path, suffix="1")
        # self.load_gating_weight(self.glayer3, (self.gating_output_dim, self.gating_hidden_dim), path, suffix="2")
        self.load_gating_weight(self.glayer1, (len(self.gating_indices), self.gating_hidden_dim), path, suffix="0")
        self.load_gating_weight(self.glayer2, (self.gating_hidden_dim, self.gating_hidden_dim), path, suffix="1")
        self.load_gating_weight(self.glayer3, (self.gating_hidden_dim, self.gating_output_dim), path, suffix="2")

        self.layer1.load_binary_weights(path, suffix='0')
        self.layer2.load_binary_weights(path, suffix='1')
        self.layer3.load_binary_weights(path, suffix='2')

    def export_discrete_weights(self, save_path):
        self.export_binary_weights(save_path)

    def load_discrete_weights(self, save_path):
        self.load_binary_weights(save_path)

    @staticmethod
    def prepare_mann_data(dataset, config_store):
        data = np.load(dataset)
        x_train = data["x"]
        y_train = data["y"]

        x_col_demarcation_ids = config_store['col_demarcation_ids'][0]
        y_col_demarcation_ids = config_store['col_demarcation_ids'][1]

        # x_pos_indices = {k: v for k, v in x_col_demarcation_ids.items() if 'pos' in k or "punch_target" in k}
        # y_pos_indices = {k: v for k, v in y_col_demarcation_ids.items() if 'pos' in k or "punch_target" in k}
        # for k, v in x_pos_indices.items():
        #     # print(k, x_train[:, v[0]:v[1]].mean())
        #     x_train[:, v[0]: v[1]] = x_train[:, v[0]: v[1]] #* 0.01
        #     # print(k, x_train[:, v[0]:v[1]].mean())
        #
        # for k, v in y_pos_indices.items():
        #     y_train[:, v[0]:v[1]] = y_train[:, v[0]:v[1]] #* 0.01

        x_mean = np.mean(x_train, axis=0)
        y_mean = np.mean(y_train, axis=0)
        x_std = np.std(x_train, axis=0)
        y_std = np.std(y_train, axis=0)

        # importance_trajectory = 1.0

        # joint_indices = config_store['joint_indices']
        # joints_ids_that_dont_matter = [val for key, val in joint_indices.items() if 'RightHand' in key][1:] + \
        #                               [val for key, val in joint_indices.items() if 'LeftHand' in key][1:]
        # joints_keys_that_matter = [k for k, v in joint_indices.items() if
        #                            v not in joints_ids_that_dont_matter]
        # joints_ids_that_matter = [val for key, val in joint_indices.items() if key in joints_keys_that_matter]
        # joint_weights = np.array([1 if val in joints_ids_that_matter else 1e-10 for val in joint_indices.values()])
        # joint_weights = np.array([1] * len(joint_indices.values()))
        # x_tr_col_indices = {k: v for k, v in x_col_demarcation_ids.items() if '_tr' in k}

        for k, v in x_col_demarcation_ids.items():
            x_std[v[0]: v[1]] = x_std[v[0]: v[1]].mean()
            # if "_tr" in k:
            #     x_std[v[0]: v[1]] *= importance_trajectory
            # if "_local" in k:
            #     x_std[v[0]: v[1]] *= (joint_weights.repeat(3))

        for k, v in y_col_demarcation_ids.items():
            y_std[v[0]: v[1]] = y_std[v[0]: v[1]].mean()
            # if "_tr" in k:
            #     y_std[v[0]: v[1]] *= importance_trajectory
            # if "_local" in k:
            #     y_std[v[0]: v[1]] *= (joint_weights.repeat(3))

        x_std[x_std == 0] = 0.01
        y_std[y_std == 0] = 0.01

        norm = {"x_mean": x_mean.tolist(),
                "y_mean": y_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_std": y_std.tolist()}

        # TODO Setup comparator to warn if input and inverse of output are the same or not

        x_train_norm = (x_train - x_mean) / x_std
        y_train_norm = (y_train - y_mean) / y_std

        raise_nan_exception(x_train_norm)
        raise_nan_exception(y_train_norm)

        return x_train_norm, y_train_norm, norm

    @staticmethod
    def train_mann(normalized_x, normalized_y, norm, gating_indices, model_path, epochs):
        """
        This constant function loads a *.npz numpy stored dataset, builds the network and trains it.
        The data is assumed to be stored as
            * "Xun": network input,
            * "Yun": network output and
            * "Pun": phase information

        Arguments:
            dataset {string} -- path to *.npz stored dataset
            target_path {string} -- path to target folder, in which networks should be stored.
            epochs {int} -- Training duration in epochs
        """
        # TODO check tensorboard to see if MANN is working correctly
        input_dim = normalized_x.shape[1]
        output_dim = normalized_y.shape[1]

        print('################################')
        print(normalized_x.shape)
        print(normalized_y.shape)

        mann_config = {"n_controls": 4, "input_dim": input_dim, "output_dim": output_dim, "dropout_rate": 0.6,
                       # TODO : maybe reduce h_dims
                       "hidden_dim": 512,

                       "gating_indices": gating_indices, "gating_hidden_dim": 32, "norm": norm}

        mann, config = InstantiateNetworkFromConfig(mann_config, MANN)

        train_config = {
            "optimizer": "Adam",
            # "learning_rate": 0.00001,
            "learning_rate": 3e-4,
            "epochs": epochs,
            # "batchsize": 32,
            "batchsize": 64,
            "loss": "mse"
        }

        TrainNetwork(mann, normalized_x, normalized_y, train_config)
        mann.export_discrete_weights(model_path)
        return mann_config

    @staticmethod
    def forward_pass(mann, x, norm, col_demarcation_ids):

        x_mean = np.array(norm['x_mean'], dtype=np.float64)
        x_std = np.array(norm['x_std'], dtype=np.float64)
        y_mean = np.array(norm['y_mean'], dtype=np.float64)
        y_std = np.array(norm['y_std'], dtype=np.float64)
        x_input = (x - x_mean) / x_std

        # print("#####################################")
        tmp = x_input.ravel()
        r_p_label = tmp[
                    col_demarcation_ids[0]['x_right_punch_labels'][0]:col_demarcation_ids[0]['x_right_punch_labels'][1]]
        l_p_label = tmp[
                    col_demarcation_ids[0]['x_left_punch_labels'][0]:col_demarcation_ids[0]['x_left_punch_labels'][1]]
        # print('rph:', r_p_label, 'lph:', l_p_label)

        y_prediction = mann(x_input)
        if np.isnan(y_prediction).any():
            raise Exception('Nans found')

        y_prediction = y_prediction * y_std + y_mean
        tmp = y_prediction.numpy().ravel()
        r_p_label = tmp[
                    col_demarcation_ids[1]['y_right_punch_labels'][0]:col_demarcation_ids[1]['y_right_punch_labels'][1]]
        l_p_label = tmp[
                    col_demarcation_ids[1]['y_left_punch_labels'][0]:col_demarcation_ids[1]['y_left_punch_labels'][1]]
        # print('rph:', r_p_label, 'lph:', l_p_label)

        if np.isnan(y_prediction).any():
            raise Exception('Nans found')

        return y_prediction


def raise_nan_exception(arr):
    if np.isnan(arr).any():
        raise Exception('Nans found in: ', np.argwhere(np.isnan(arr)))
