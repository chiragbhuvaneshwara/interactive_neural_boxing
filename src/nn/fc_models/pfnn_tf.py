from .layers.layers_tf import TF_PFNN_Layer
from .fc_networks import FCNetwork

import tensorflow as tf
import numpy as np
import sys, os, datetime, json

class PFNN(FCNetwork):
	"""

	Implementation of phase-functioned neural networks using tensorflow backend. 

	This class is a realization of FCNetwork. 

	Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

	@author: Janis
	"""


	def __init__(self, input_size, output_size, hidden_size, norm, batch_size = 32, layers = [], dropout = 0.7):
		"""

		Implementation of phase-functioned neural networks using tensorflow backend. 

		This class is a realization of FCNetwork. 

		Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

		Arguments:
			input_size {int} -- size of the input vector
			output_size {int} -- size of the output vector
			hidden_size {int} -- size of the hidden layers
			norm {map} -- map, containing the normalization information: Xmean, Ymean, Xstd, Ystd. 
		"""

		super().__init__(input_size, output_size, hidden_size, norm)

		self.batch_size = batch_size
		self.dropout = dropout

		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name="InputString")
		self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_size], name="OutputString")
		self.p = tf.placeholder(tf.float32, shape=[self.batch_size], name = "Phase")

		if len(layers) != 3:
			l0 = TF_PFNN_Layer((hidden_size, input_size), elu_operator = tf.nn.elu, name="Layer0")
			l1 = TF_PFNN_Layer((hidden_size, hidden_size), elu_operator =tf.nn.elu, name="Layer1")
			l2 = TF_PFNN_Layer((output_size, hidden_size), name="Layer2")
		else:
			l0, l1, l2 = layers[0], layers[1], layers[2]


		self.add_layer(l0)
		self.add_layer(l1)
		self.add_layer(l2)

		self.train_step = None
		self.cost_function = None

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
			if not nlayers == 3:
				print("ERROR: layers not matching. Stored: " + nlayers)
				return

			layers = [TF_PFNN_Layer.load([store["layer_0"], tf.nn.elu]), 
						TF_PFNN_Layer.load([store["layer_1"], tf.nn.elu]),
						TF_PFNN_Layer.load([store["layer_2"], None])]
			pfnn = PFNN(input_size, output_size, hidden_size, norm, batch_size = 1, layers=layers, dropout = 0.999)

			return pfnn
		
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
		#p = tf.gather(self.x, [:, -1])
		#net_in = self.x[:, :-1]
		dropout_prob = tf.constant(self.dropout, tf.float32, name="dropoutProb")

		params = [self.x, self.p, dropout_prob]
		print("input shape: ", self.x.shape)
		params = super().build_tf_graph(params)
		self.network_output = tf.identity(params[0], name="networkOut")
		output = self.network_output
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
		
		opt = tf.train.AdamOptimizer(learning_rate = 0.0001)
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
		self.build_tf_graph([])

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())

			self.summary_writer = tf.summary.FileWriter("../logs/", sess.graph)
			for l in self.layers:
				l.sess = sess

			I = np.arange(len(X))
			print("processing input: ", X[I].shape)
			for e in range(epochs):
				print("starting epoch %d"%e)
				# randomize network input
				np.random.shuffle(I)
				input,output = X[I], Y[I]
				
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
					p = x[:,-1]
					x = x[:,:-1]
					y = output[i:i + self.batch_size]
					#y = y_data.reshape(self.batch_size, self.output_shape)
					merged, _, loss_value, acc = sess.run([self.merged, self.train_step, self.cost_function, self.accuracy], feed_dict={self.x: x, self.y: y, self.p: p})
					self.summary_writer.add_summary(merged, i + e * len(input))
					step_counter += 1
					if i > 0 and (i % 20) == 0:
						sys.stdout.write(
							'\r[Epoch %3i] % 3.1f%% est. remaining time: %i min, %.5f loss, %.5f acc' % (
						e, 100 * i / len(input), ((datetime.datetime.now() - start_time).total_seconds() / i * (len(input) - i)) / 60, loss_value, acc))
						sys.stdout.flush()
					accuracies.append(acc)
					losses.append(loss_value)
				print("end of epoch: ")
				print("duration: ",  (datetime.datetime.now() - start_time).total_seconds() / 60, "min")
				print("average loss: ", np.mean(losses))
				print("average accuracy: ", np.mean(accuracies))
				print("")
				
				# save epoch: 
				#os.makedirs(target_path + "/epoch_%d"%e, exist_ok=True)
				self.store(target_path + "/epoch_%d.json"%e)
	
	def start_tf(self):
		self.build_tf_graph([])
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())
	
	def forward_pass(self, params):
		# normalize: 
		params[0] = np.array(params[0])
		if len(params[0]) == 0:
			params[0] = np.array(self.norm["Xmean"])
		params[0] = (params[0] - self.norm["Xmean"]) / self.norm["Xstd"]

		
		out = self.sess.run([self.network_output], feed_dict={self.x: [params[0]], self.p: [params[1]]})
		out = np.array(out[0][0])
		out = (out * self.norm["Ystd"]) + self.norm["Ymean"]
		return [out, params[1]]

	def from_file(dataset, target_path, epochs, config_store):
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
		P = data["Pun"]

		Xmean = np.mean(X, axis=0)
		Ymean = np.mean(Y, axis=0)
		Xstd = np.std(X, axis=0)
		Ystd = np.std(Y, axis=0)

		w = (60 * 2) // 10
		j = config_store["numJoints"]
		gaits = config_store["n_gaits"]

		joint_weights = np.array([
			1,
			1e-10, 1, 1, 1, 1,
			1e-10, 1, 1, 1, 1,
			1e-10, 1, 1,
			1e-10, 1, 1,
			1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
			1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10])

		Xstd[w * 0:w * 1] = Xstd[w * 0:w * 1].mean()  # Trajectory Pos X
		Xstd[w * 1:w * 2] = Xstd[w * 1:w * 2].mean()  # Trajectory Pos Y
		Xstd[w * 2:w * 3] = Xstd[w * 2:w * 3].mean()  # Trajectory Directions X
		Xstd[w * 3:w * 4] = Xstd[w * 3:w * 4].mean()  # Trajectory Directions Y
		Xstd[w * 4:w * 4 + w * gaits] = Xstd[w * 4:w * 4 + w * gaits].mean()
		start = w * (4 + gaits)

		Xstd[start + j * 3 * 0:start + j * 3 * 1] = Xstd[start + j * 3 * 0:start + j * 3 * 1].mean() / (joint_weights.repeat(3) * 0.1)  # Pos
		Xstd[start + j * 3 * 1:start + j * 3 * 2] = Xstd[start + j * 3 * 1:start + j * 3 * 2].mean() / (joint_weights.repeat(3) * 0.1)  # Vel
		#Xstd[w * start + j * 3 * 2:w * start + j * (3 * 2) + (j)] = Xstd[w * start + j * 3 * 2:w * start + j * (3 * 2) + (j)].mean() / (joint_weights * 0.1)  # twists
		#start = start + j * 3 * 2
		#Xstd[start:] = Xstd[start:].mean()
		print("mean and std. dev of translation vel: ", Ymean[0:3], Ystd[0:3])
		Ystd[0:1] = Ystd[0:1].mean()
		Ystd[1:2] = Ystd[1:2].mean()  # Translational Velocity
		Ystd[2:3] = Ystd[2:3].mean()  # Rotational Velocity
		Ystd[3:4] = Ystd[3:4].mean()  # Change in Phase
		if config_store["use_footcontacts"]:
			print("using Footcontacts")
			Ystd[4:8] = Ystd[4:8].mean() # foot contacts
			start = 8
		else:
			start = 4
		
		Ystd[start + (w // 2) * 0:start + (w // 2) * 1] = Ystd[start + (w // 2) * 0:start + (w // 2) * 1].mean()  # Trajectory Future Positions X
		Ystd[start + (w // 2) * 1:start + (w // 2) * 2] = Ystd[start + (w // 2) * 1:start + (w // 2) * 2].mean()  # Trajectory Future Positions Y
		Ystd[start + (w // 2) * 2:start + (w // 2) * 3] = Ystd[start + (w // 2) * 2:start + (w // 2) * 3].mean()  # Trajectory Future Directions X
		Ystd[start + (w // 2) * 3:start + (w // 2) * 4] = Ystd[start + (w // 2) * 3:start + (w // 2) * 4].mean()  # Trajectory Future Directions Y
		
		start = start + (w // 2) * 4
		Ystd[start + j * 3 * 0:start + j * 3 * 1] = Ystd[start + j * 3 * 0:start + j * 3 * 1].mean()  # Pos
		Ystd[start + j * 3 * 1:start + j * 3 * 2] = Ystd[start + j * 3 * 1:start + j * 3 * 2].mean()  # Vel

		#Xstd[Xstd == 0] = 1.0
		#Ystd[Ystd == 0] = 1.0
		norm = {"Xmean": Xmean.tolist(), 
				"Ymean":Ymean.tolist(), 
				"Xstd":Xstd.tolist(), 
				"Ystd":Ystd.tolist()}

		X = (X - Xmean) / Xstd
		Y = (Y - Ymean) / Ystd
		
		#np.savez("%s/Ymean"%target_path, Ymean)
		#np.savez("%s/Xmean"%target_path, Xmean)
		#np.savez("%s/Ystd"%target_path, Ystd)
		#np.savez("%s/Xstd"%target_path, Xstd)

		input_dim = X.shape[1]
		output_dim = Y.shape[1]

		X = np.concatenate([X, P[:, np.newaxis]], axis=-1)


		pfnn = PFNN(input_dim, output_dim, 512, norm)
		pfnn.train(X, Y, epochs, target_path)