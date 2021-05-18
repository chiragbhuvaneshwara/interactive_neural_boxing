import numpy as np
import tensorflow as tf


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
            r = w * e
            return tf.reduce_sum(r, axis=1)

    @tf.function
    def gating_network(self, inputs, training=None):
        if training:
            H0 = tf.nn.dropout(inputs, self.dropout_prob)
        else:
            H0 = inputs

        H1 = tf.matmul(self.weight_knots[0], H0) + self.bias_knots[0]
        H1 = tf.nn.elu(H1)
        if not training is None:
            H1 = tf.nn.dropout(H1, self.dropout_prob)

        H2 = tf.matmul(self.weight_knots[1], H1) + self.bias_knots[1]
        H2 = tf.nn.elu(H2)
        if not training is None:
            H2 = tf.nn.dropout(H2, self.dropout_prob)

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

        H1 = tf.matmul(w0, H0) + b0
        H1 = tf.nn.elu(H1)
        if not training is None:
            H1 = tf.nn.dropout(H1, self.dropout_prob)

        w1 = self.interpolate(self.weight_knots[4], expert_weights)
        b1 = self.interpolate(self.bias_knots[4], expert_weights)

        H2 = tf.matmul(w1, H1) + b1
        H2 = tf.nn.elu(H2)
        if not training is None:
            H2 = tf.nn.dropout(H2, self.dropout_prob)

        w2 = self.interpolate(self.weight_knots[5], expert_weights)
        b2 = self.interpolate(self.bias_knots[5], expert_weights)

        H3 = tf.matmul(w2, H2) + b2

        return H3

    @tf.function
    def call(self, inputs, training=None):
        expert_input = tf.expand_dims(tf.gather(inputs, self.gating_indices, axis=1), -1)
        motion_input = tf.expand_dims(inputs, -1)
        expert_weights = self.gating_network(expert_input, training)[..., 0]
        output = self.motion_network(motion_input, expert_weights, training)[..., 0]

        return tf.concat([output, expert_weights], axis=-1)

    def get_summary(self):
        network_summary = {
            "input_size_motion_network": self.input_size,
            "output_size_motion_network": self.output_size,
            "hidden_size_motion_network": self.hidden_size,
            "input_size_gating_network": self.gating_input,
            "hidden_size_gating_network": self.gating_hidden,
            "gating_indices": self.gating_indices,
            "expert_nodes": self.expert_nodes,
            "batch_size": self.batch_size,
            "dropout_prob": self.dropout_prob,
        }

        return network_summary

    @staticmethod
    def forward_pass(mann, x, norm, col_demarcation_ids):

        # TODO make norm save and extract functions
        x_mean = np.array(norm['x_mean'], dtype=np.float64)
        x_std = np.array(norm['x_std'], dtype=np.float64)
        y_mean = np.array(norm['y_mean'], dtype=np.float64)
        y_std = np.array(norm['y_std'], dtype=np.float64)
        x_input = (x - x_mean) / x_std

        y_prediction = np.array(mann(x_input)).ravel()
        if np.isnan(y_prediction).any():
            raise Exception('Nans found')

        gating_weights = y_prediction[-6:]
        y_prediction = y_prediction[:-6]
        y_prediction = np.array(y_prediction * y_std + y_mean).ravel()

        if np.isnan(y_prediction).any():
            raise Exception('Nans found')

        return y_prediction
