from .layers.layers_tf import TF_MANN_Layer
from .layers.layers_tf import TF_FCLayer
from .fc_networks import FCNetwork

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import sys, os, datetime, json
import pandas as pd

class MANN(FCNetwork):
    """

    Implementation of mode-adaptive neural networks using tensorflow backend. 

    This class is a realization of FCNetwork. 

    Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

    @author: Janis and Usama (for changes w.r.t MANN)
    """

    def __init__(self, input_size, output_size, hidden_size, norm, batch_size=32, layers=[], dropout=0.7,
                 gating_indices=[0, 1, 2]):
        """

        Implementation of mode-adaptive neural networks using tensorflow backend. 

        This class is a realization of FCNetwork. 

        Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

        Arguments:
            input_size {int} -- size of the input vector
            output_size {int} -- size of the output vector
            hidden_size {int} -- size of the hidden layers
            norm {map} -- map, containing the normalization information: Xmean, Ymean, Xstd, Ystd. 
        """

        super().__init__(input_size, output_size, hidden_size, norm)
        print("Init MANN TF")

        self.batch_size = batch_size
        self.dropout = dropout

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name="InputString")
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_size], name="OutputString")
        # self.p = tf.placeholder(tf.float32, shape=[self.batch_size], name = "Phase")

        self.counter = tf.placeholder(tf.int32, name="step_counter")

        self.gatingnet = FCNetwork(len(gating_indices), 4, 32, norm)
        self.gating_indices = gating_indices

        GL0 = TF_FCLayer((32, len(gating_indices)), None, None, elu_operator=tf.nn.elu, name="GLayer0")
        GL1 = TF_FCLayer((32, 32), None, None, elu_operator=tf.nn.elu, name="GLayer1")
        GL2 = TF_FCLayer((4, 32), None, None, name="GLayer2")

        self.gatingnet.add_layer(GL0)
        self.gatingnet.add_layer(GL1)
        self.gatingnet.add_layer(GL2)

        if len(layers) != 3:
            l0 = TF_MANN_Layer((hidden_size, input_size), elu_operator=tf.nn.elu, name="MPLayer0")
            l1 = TF_MANN_Layer((hidden_size, hidden_size), elu_operator=tf.nn.elu, name="MPLayer1")
            l2 = TF_MANN_Layer((output_size, hidden_size), name="MPLayer2")
        else:
            l0, l1, l2 = layers[0], layers[1], layers[2]

        self.add_layer(l0)
        self.add_layer(l1)
        self.add_layer(l2)

        self.train_step = None
        self.cost_function = None
        self.first_decay_steps = 0

    def store(self, target_file):
        """

        Stores the whole network in a json file. 
        
        Arguments:
            target_file {string} -- path to the target *.json file. 
        """
        store = {"nlayers": (len(self.layers)),
                 "input_size": (self.input_size),
                 "output_size": (self.output_size),
                 "hidden_size": (self.hidden_size),
                 "norm": self.norm,
                 "gatingIndices": self.gating_indices}

        for l in range(len(self.layers)):
            store["layer_%d" % l] = self.layers[l].store()

        with open(target_file, "w") as f:
            json.dump(store, f)

    def load(target_file):
        with open(target_file, "r") as f:
            store = json.load(f)
            # store = {"nlayers":(len(self.layers)),
            # 		"input_size":(self.input_size),
            # 		"output_size":(self.output_size),
            # 		"hidden_size":(self.hidden_size), 
            # 		"norm":self.norm}
            input_size = store["input_size"]
            output_size = store["output_size"]
            hidden_size = store["hidden_size"]
            norm = store["norm"]
            nlayers = store["nlayers"]
            gating_indices = store["gatingIndices"]
            if not nlayers == 3:
                print("ERROR: layers not matching. Stored: " + nlayers)
                return

            layers = [TF_MANN_Layer.load([store["layer_0"], tf.nn.elu]),
                      TF_MANN_Layer.load([store["layer_1"], tf.nn.elu]),
                      TF_MANN_Layer.load([store["layer_2"], None])]
            mann = MANN(input_size, output_size, hidden_size, norm, batch_size=1, layers=layers, dropout=1,
                        gating_indices=gating_indices)
            mann.configstore = store
            return mann

        # for l in range(len(self.layers)):
        # 	store["layer_%d"%l] = self.layers[l].store()

        # with open(target_file, "w") as f:
        # 	json.dump(store, f)

    def build_tf_graph(self, params):
        """
        Builds the network graph for tensorflow. This should not be called directly, but is used in the training function. 
        
        Arguments:
            params {list} -- Unused.
        """
        # p = tf.gather(self.x, [:, -1])
        # net_in = self.x[:, :-1]
        dropout_prob = tf.constant(self.dropout, tf.float32, name="dropoutProb")

        # self.x[:,self.gating_indices]
        gating_input = tf.gather(self.x, self.gating_indices, axis=1)
        print("gating input: ", gating_input)

        params = [gating_input, 0, dropout_prob]
        print("input shape: ", self.x.shape)

        out_gating = self.gatingnet.build_tf_graph(params)
        # softmax on out_gating[0]
        sft_out_gating = tf.nn.softmax(out_gating[0], name="gnetworkOut")
        params = [self.x, sft_out_gating, dropout_prob]

        params = super().build_tf_graph(params)

        self.network_output = tf.identity(params[0], name="mpnetworkOut")
        output = tf.reshape(self.network_output, (self.network_output.shape[0], self.network_output.shape[1]))
        print("output shape: ", output.shape)

        tf.global_variables_initializer()

        with tf.name_scope("costAcc") as scope:
            self.cost_function = tf.reduce_mean((self.y - output) ** 2) + 0.01 * (
                    tf.reduce_mean(tf.abs(self.layers[0].weight)) +
                    tf.reduce_mean(tf.abs(self.layers[1].weight)) +
                    tf.reduce_mean(tf.abs(self.layers[2].weight))
            )
            self.accuracy = tf.reduce_mean((self.y - output) ** 2)

        tf.summary.scalar("cost_function", self.cost_function)
        tf.summary.scalar("training accuracy", self.accuracy)

        self.lr_decayed = tf.train.cosine_decay_restarts(0.0001, self.counter, self.first_decay_steps)

        opt = tf.train.AdamOptimizer(self.lr_decayed)  # learning_rate = 0.0001
        var_list = []
        for l in self.layers:
            var_list.append(l.weight)
            var_list.append(l.bias)
        self.train_step = opt.minimize(self.cost_function, var_list=var_list)
        self.merged = tf.summary.merge_all()

    def train(self, X, Y, epochs, target_path):
        """

        Trains the neural network. The tensorflow graph is build within this function, so no further requirements are necessary.
        
        Arguments:
            X {np.array} -- Numpy array containing the input data (n_frames, x_dim)
            Y {np.array} -- Numpy array containing the output data (n_frames, y_dim)
            epochs {int} -- Training duration in epochs
            target_path {string} -- Path to a folder, where the network iterations should be stored. 
        """
        self.n_frames = len(X)
        self.first_decay_steps = self.n_frames // self.batch_size * 10

        self.build_tf_graph([])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            self.summary_writer = tf.summary.FileWriter(target_path, sess.graph)
            for l in self.layers:
                l.sess = sess

            I = np.arange(len(X))
            print("processing input: ", X[I].shape)
            counter = np.array([0], dtype=np.int32)
            for e in range(epochs):
                print("starting epoch %d" % e)
                # randomize network input
                np.random.shuffle(I)
                input, output = X[I], Y[I]

                start_time = datetime.datetime.now()
                accuracies = []
                losses = []

                batchind = np.arange(len(input) // self.batch_size)  # start batches from these indices.
                np.random.shuffle(batchind)

                step_counter = 0

                for i in range(0, len(input), self.batch_size):
                    if (i + self.batch_size) >= len(input):
                        break
                    x = input[i:(i + self.batch_size), :]
                    p = x[:, -1]
                    # x = x[:,:-1]
                    y = output[i:i + self.batch_size]
                    # y = y_data.reshape(self.batch_size, self.output_shape)
                    # merged, _, loss_value, acc = sess.run([self.merged, self.train_step, self.cost_function, self.accuracy], feed_dict={self.x: x, self.y: y, self.p: p, self.counter: counter[0]})
                    merged, _, loss_value, acc = sess.run(
                        [self.merged, self.train_step, self.cost_function, self.accuracy],
                        feed_dict={self.x: x, self.y: y, self.counter: counter[0]})
                    # sess.run(self.counter.assign_add(1))
                    self.summary_writer.add_summary(merged, i + e * len(input))
                    step_counter += 1
                    counter[0] += 1
                    if i > 0 and (i % 20) == 0:
                        sys.stdout.write(
                            '\r[Epoch %3i] % 3.1f%% est. remaining time: %i min, %.5f loss, %.5f acc, lr * 1000 %.6f, c: %i' % (
                                e, 100 * i / len(input),
                                ((datetime.datetime.now() - start_time).total_seconds() / i * (len(input) - i)) / 60,
                                loss_value, acc, sess.run(self.lr_decayed, feed_dict={self.counter: counter[0]}) * 1000,
                                counter[0]))
                        sys.stdout.flush()
                    accuracies.append(acc)
                    losses.append(loss_value)
                    print(loss_value)

                print("end of epoch: ")
                print("duration: ", (datetime.datetime.now() - start_time).total_seconds() / 60, "min")
                print("average loss: ", np.mean(losses))
                print("average accuracy: ", np.mean(accuracies))
                print("")

                # save epoch: 
                # os.makedirs(target_path + "/epoch_%d"%e, exist_ok=True)
                self.store(target_path + "/epoch_%d.json" % e)

    def start_tf(self):
        self.build_tf_graph([])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def forward_pass(self, params):
        # normalize: 
        # params = np.array(params).reshape(1,362)
        params = params.ravel()
        if len(params) == 0:
            params = np.array(self.norm["Xmean"])
        # params = params.reshape(1, 362)
        params = params.reshape(1, self.input_size)
        # params = params.ravel()
        params = (params - self.norm["Xmean"]) / self.norm["Xstd"]

        out = self.sess.run(self.network_output, feed_dict={self.x: params})
        # out = np.array(out).reshape(1,362)
        out = np.array(out).ravel()
        out = out.reshape(1, self.output_size)
        out = (out * self.norm["Ystd"]) + self.norm["Ymean"]
        out = out.ravel()
        return out

    @staticmethod
    def from_file(dataset, target_path, epochs, config_store, gating_indices=[0, 1, 2]):
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
        data = np.load(dataset)
        X = data["Xun"]
        Y = data["Yun"]
        # P = data["Pun"]

        Xmean = np.mean(X, axis=0)
        Ymean = np.mean(Y, axis=0)
        Xstd = np.std(X, axis=0)
        Ystd = np.std(Y, axis=0)

        # w = (60 * 2) // 10
        # j = config_store["numJoints"]
        # gaits = config_store["n_gaits"]

        joint_indices = config_store['joint_indices']
        joints_ids_that_dont_matter = [val for key, val in joint_indices.items() if 'RightHand' in key][1:] + [val for key, val in joint_indices.items() if 'LeftHand' in key][1:]
        joints_keys_that_matter = [k for k, v in joint_indices.items() if
                                   v not in joints_ids_that_dont_matter]
        joints_ids_that_matter = [val for key, val in joint_indices.items() if key in joints_keys_that_matter]
        joint_weights = np.array([1 if val in joints_ids_that_matter else 1e-10 for val in joint_indices.values()])

        # joint_weights = np.array([
        #     1,
        #     1e-10, 1, 1, 1, 1,
        #     1e-10, 1, 1, 1, 1,
        #     1e-10, 1, 1,
        #     1e-10, 1, 1,
        #     1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
        #     1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10])

        x_col_indices = config_store['col_indices'][0]
        y_col_indices = config_store['col_indices'][1]

        x_tr_col_indices = {k:v for k,v in x_col_indices.items() if '_tr' in k}
        y_tr_col_indices = {k:v for k,v in y_col_indices.items() if '_tr' in k}

        for k,v in x_tr_col_indices.items():
            Xstd[v[0]: v[1]] = Xstd[v[0]: v[1]].mean()


        # Xstd[w * 0:w * 1] = Xstd[w * 0:w * 1].mean()  # Trajectory Pos X
        # Xstd[w * 1:w * 2] = Xstd[w * 1:w * 2].mean()  # Trajectory Pos Y
        # Xstd[w * 2:w * 3] = Xstd[w * 2:w * 3].mean()  # Trajectory Directions X
        # Xstd[w * 3:w * 4] = Xstd[w * 3:w * 4].mean()  # Trajectory Directions Y
        # Xstd[w * 4:w * 4 + w * gaits] = Xstd[w * 4:w * 4 + w * gaits].mean()

        x_local_col_indices = {k:v for k,v in x_col_indices.items() if '_local' in k}
        y_local_col_indices = {k:v for k,v in y_col_indices.items() if '_local' in k}

        for k, v in x_local_col_indices.items():
            Xstd[v[0]: v[1]] = Xstd[v[0]: v[1]].mean() / (joint_weights.repeat(3))    # * 0.1)

        # start = w * (4 + gaits)
        # Xstd[start + j * 3 * 0:start + j * 3 * 1] = Xstd[
        #                                             start + j * 3 * 0:start + j * 3 * 1].mean()  # / (joint_weights.repeat(3) * 0.1)  # Pos
        # Xstd[start + j * 3 * 1:start + j * 3 * 2] = Xstd[
        #                                             start + j * 3 * 1:start + j * 3 * 2].mean()  # / (joint_weights.repeat(3) * 0.1)  # Vel
        # # Xstd[w * start + j * 3 * 2:w * start + j * (3 * 2) + (j)] = Xstd[w * start + j * 3 * 2:w * start + j * (3 * 2) + (j)].mean() / (joint_weights * 0.1)  # twists
        # # start = start + j * 3 * 2
        # # Xstd[start:] = Xstd[start:].mean()


        print("mean and std. dev of translation vel: ", Ymean[0:3], Ystd[0:3])
        importance_trajectory = 1.0

        y_col_other_indices = {k:v for k,v in y_col_indices.items() if ('_tr' not in k) or ('_local' not in k)}

        r_vel_indices = y_col_other_indices['y_root_velocity']
        Ystd[r_vel_indices[0]:r_vel_indices[1]] = Ystd[r_vel_indices[0]:r_vel_indices[1]].mean() / importance_trajectory
        r_new_forward_indices = y_col_other_indices['y_root_new_forward']
        Ystd[r_new_forward_indices[0]:r_new_forward_indices[1]] = Ystd[r_new_forward_indices[0]:r_new_forward_indices[1]].mean() / importance_trajectory
        p_dphase_indices = y_col_other_indices['y_punch_dphase']
        Ystd[p_dphase_indices[0]:p_dphase_indices[0]+1] = Ystd[p_dphase_indices[0]:p_dphase_indices[0]+1].mean() / importance_trajectory
        Ystd[p_dphase_indices[-1]-1:p_dphase_indices[-1]] = Ystd[p_dphase_indices[0]:p_dphase_indices[0]+1].mean() / importance_trajectory

        # Ystd[0:1] = Ystd[0:1].mean() / importance_trajectory
        # Ystd[1:2] = Ystd[1:2].mean() / importance_trajectory  # Translational Velocity
        # Ystd[2:3] = Ystd[2:3].mean() / importance_trajectory  # Rotational Velocity / Dir x
        # Ystd[3:4] = Ystd[3:4].mean() / importance_trajectory  # Change in Phase  / Dir y
        # Ystd[4:5] = Ystd[4:5].mean() / importance_trajectory  # Change in Phase
        # start = 5

        if config_store["use_footcontacts"]:
            print("using Footcontacts")

            foot_indices = y_col_other_indices['y_foot_contacts']
            Ystd[foot_indices[0]:foot_indices[1]] = Ystd[foot_indices[0]:foot_indices[1]].mean()    # foot contacts

            # Ystd[start:start + 4] = Ystd[start:start + 4].mean()  # foot contacts
            # start += 4

        for k,v in y_tr_col_indices.items():
            Ystd[v[0]: v[1]] = Ystd[v[0]: v[1]].mean()

        # Ystd[start + (w // 2) * 0:start + (w // 2) * 1] = Ystd[start + (w // 2) * 0:start + (
        #         w // 2) * 1].mean()  # Trajectory Future Positions X
        # Ystd[start + (w // 2) * 1:start + (w // 2) * 2] = Ystd[start + (w // 2) * 1:start + (
        #         w // 2) * 2].mean()  # Trajectory Future Positions Y
        # Ystd[start + (w // 2) * 2:start + (w // 2) * 3] = Ystd[start + (w // 2) * 2:start + (
        #         w // 2) * 3].mean()  # Trajectory Future Directions X
        # Ystd[start + (w // 2) * 3:start + (w // 2) * 4] = Ystd[start + (w // 2) * 3:start + (
        #         w // 2) * 4].mean()  # Trajectory Future Directions Y

        for k, v in y_local_col_indices.items():
            Ystd[v[0]: v[1]] = Ystd[v[0]: v[1]].mean() / (joint_weights.repeat(3))    # * 0.1)

        # start = start + (w // 2) * 4
        # Ystd[start + j * 3 * 0:start + j * 3 * 1] = Ystd[start + j * 3 * 0:start + j * 3 * 1].mean()  # Pos
        # Ystd[start + j * 3 * 1:start + j * 3 * 2] = Ystd[start + j * 3 * 1:start + j * 3 * 2].mean()  # Vel

        Xstd[Xstd == 0] = 1.0
        Ystd[Ystd == 0] = 1.0

        norm = {"Xmean": Xmean.tolist(),
                "Ymean": Ymean.tolist(),
                "Xstd": Xstd.tolist(),
                "Ystd": Ystd.tolist()}

        X = (X - Xmean) / Xstd
        Y = (Y - Ymean) / Ystd

        # np.savez("%s/Ymean"%target_path, Ymean)
        # np.savez("%s/Xmean"%target_path, Xmean)
        # np.savez("%s/Ystd"%target_path, Ystd)
        # np.savez("%s/Xstd"%target_path, Xstd)

        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        # X = np.concatenate([X, P[:, np.newaxis]], axis=-1)

        print('################################')
        print(X.shape)
        print(X[1])
        print(Y.shape)
        print(Y[1])
        mann = MANN(input_dim, output_dim, 512, norm, gating_indices=gating_indices)
        mann.train(X, Y, epochs, target_path)
