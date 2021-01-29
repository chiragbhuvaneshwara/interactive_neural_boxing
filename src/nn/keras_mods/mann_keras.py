import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

from src.nn.keras_mods.network_instantiater import TrainNetwork, InstantiateNetworkFromConfig
from sklearn import preprocessing

from zipfile import ZipFile

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
            path {[type]} -- path to zip file

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
            path {[type]} -- path to zip file

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
        X = data["Xun"]
        Y = data["Yun"]

        x_col_indices = config_store['col_indices'][0]
        y_col_indices = config_store['col_indices'][1]

        x_pos_indices = {k: v for k, v in x_col_indices.items() if 'pos' in k or "punch_target" in k}
        y_pos_indices = {k: v for k, v in y_col_indices.items() if 'pos' in k or "punch_target" in k}
        for k, v in x_pos_indices.items():
            print(k, X[:, v[0]:v[1]].mean())
            X[:, v[0]: v[1]] = X[:, v[0]: v[1]] * 0.01
            print(k, X[:, v[0]:v[1]].mean())

        for k, v in y_pos_indices.items():
            Y[:, v[0]:v[1]] = Y[:, v[0]:v[1]] * 0.01

        Xmean = np.mean(X, axis=0)
        Ymean = np.mean(Y, axis=0)
        Xstd = np.std(X, axis=0)
        Ystd = np.std(Y, axis=0)

        # joint_indices = config_store['joint_indices']
        # joints_ids_that_dont_matter = [val for key, val in joint_indices.items() if 'RightHand' in key][1:] + \
        #                               [val for key, val in joint_indices.items() if 'LeftHand' in key][1:]
        # joints_keys_that_matter = [k for k, v in joint_indices.items() if
        #                            v not in joints_ids_that_dont_matter]
        # joints_ids_that_matter = [val for key, val in joint_indices.items() if key in joints_keys_that_matter]
        # joint_weights = np.array([1 if val in joints_ids_that_matter else 1e-10 for val in joint_indices.values()])
        # joint_weights = np.array([1] * len(joint_indices.values()))

        x_tr_col_indices = {k: v for k, v in x_col_indices.items() if '_tr' in k}
        y_tr_col_indices = {k: v for k, v in y_col_indices.items() if '_tr' in k}

        for k, v in x_tr_col_indices.items():
            Xstd[v[0]: v[1]] = Xstd[v[0]: v[1]].mean()

        x_local_col_indices = {k: v for k, v in x_col_indices.items() if '_local' in k}
        y_local_col_indices = {k: v for k, v in y_col_indices.items() if '_local' in k}

        for k, v in x_local_col_indices.items():
            Xstd[v[0]: v[1]] = Xstd[v[0]: v[1]].mean() # * (joint_weights.repeat(3))  # * 0.1)

        importance_trajectory = 1.0

        y_col_other_indices = {k: v for k, v in y_col_indices.items() if ('_tr' not in k) or ('_local' not in k)}

        r_vel_indices = y_col_other_indices['y_root_velocity']
        Ystd[r_vel_indices[0]:r_vel_indices[1]] = Ystd[r_vel_indices[0]:r_vel_indices[1]].mean() # / importance_trajectory

        r_new_forward_indices = y_col_other_indices['y_root_new_forward']
        Ystd[r_new_forward_indices[0]:r_new_forward_indices[1]] = Ystd[r_new_forward_indices[0]:r_new_forward_indices[
            1]].mean() # / importance_trajectory

        # p_dphase_indices = y_col_other_indices['y_punch_dphase']
        p_dphase_indices = y_col_other_indices['y_punch_phase']
        Ystd[p_dphase_indices[0]:p_dphase_indices[0] + 1] = \
            Ystd[p_dphase_indices[0]:p_dphase_indices[0] + 1].mean() # / importance_trajectory
        Ystd[p_dphase_indices[-1] - 1:p_dphase_indices[-1]] = \
            Ystd[p_dphase_indices[0]:p_dphase_indices[0] + 1].mean() # / importance_trajectory

        if config_store["use_footcontacts"]:
            print("using Footcontacts")

            foot_indices = y_col_other_indices['y_foot_contacts']
            Ystd[foot_indices[0]:foot_indices[1]] = Ystd[foot_indices[0]:foot_indices[1]].mean()  # foot contacts

        for k, v in y_tr_col_indices.items():
            Ystd[v[0]: v[1]] = Ystd[v[0]: v[1]].mean()

        for k, v in y_local_col_indices.items():
            Ystd[v[0]: v[1]] = Ystd[v[0]: v[1]].mean() #* (joint_weights.repeat(3))  # * 0.1)

        Xstd[Xstd == 0] = 1.0
        Ystd[Ystd == 0] = 1.0


        norm = {"Xmean": Xmean.tolist(),
                "Ymean": Ymean.tolist(),
                "Xstd": Xstd.tolist(),
                "Ystd": Ystd.tolist()}

        # eps = 1e-100
        # X = (X - Xmean) / (Xstd + eps)
        # Y = (Y - Ymean) / (Ystd + eps)

        #todo which vals are 0

        X = (X - Xmean) / Xstd
        Y = (Y - Ymean) / Ystd
        print(X.mean())
        print(Y.mean())

        raise_nan_exception(X)
        raise_nan_exception(Y)

        return X, Y, norm

    @staticmethod
    def train_mann(normalized_X, normalized_Y, norm, gating_indices, model_path, epochs):
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
        #TODO check tensorboard to see if MANN is working correctly
        input_dim = normalized_X.shape[1]
        output_dim = normalized_Y.shape[1]

        print('################################')
        print(normalized_X.shape)
        print(normalized_Y.shape)

        mann_config = {"n_controls": 4, "input_dim": input_dim, "output_dim": output_dim, "dropout_rate": 0.6,
                       # TODO : maybe reduce h_dims
                       "hidden_dim": 512,

                       "gating_indices": gating_indices, "gating_hidden_dim": 32, "norm": norm}

        mann, config = InstantiateNetworkFromConfig(mann_config, MANN)

        train_config = {
            "optimizer": "Adam",
            "learning_rate": 0.00001,
            # "learning_rate": 3e-4,
            #TODO 100 epochs
            "epochs": epochs,
            "batchsize": 32,
            # "batchsize": 64,
            "loss": "mse"
        }

        TrainNetwork(mann, normalized_X, normalized_Y, train_config)
        mann.export_discrete_weights(model_path)
        return mann_config

    @staticmethod
    def forward_pass(mann, X, norm):

        Xmean = np.array(norm['Xmean'], dtype=np.float64)
        Xstd = np.array(norm['Xstd'], dtype=np.float64)
        Ymean = np.array(norm['Ymean'], dtype=np.float64)
        Ystd = np.array(norm['Ystd'], dtype=np.float64)
        # eps = 1e-100
        # input = (X - Xmean) / (Xstd + eps)
        input = (X - Xmean) / Xstd

        # Using the standardized input is leading to NaNs
        Y_prediction = mann(input)
        if np.isnan(Y_prediction).any():
            raise Exception('Nans found')

        # Y_prediction = mann(X)
        # if np.isnan(Y_prediction).any():
        #     raise Exception('Nans found')

        # Y_prediction = Y_prediction.numpy()
        Y_prediction = Y_prediction * Ystd + Ymean
        # print(Y_prediction.numpy()[4:6])

        if np.isnan(Y_prediction).any():
            raise Exception('Nans found')

        return Y_prediction


def raise_nan_exception(arr):
    if np.isnan(arr).any():
        raise Exception('Nans found in: ', np.argwhere(np.isnan(arr)))
