from .layers.layers_tf import TF_PFNN_Layer
from .fc_networks import FCNetwork

import tensorflow as tf
import numpy as np
import sys, os, datetime

class PFNN(FCNetwork):
	"""

	Implementation of phase-functioned neural networks using tensorflow backend. 

	This class is a realization of FCNetwork. 

	Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

	@author: Janis
	"""


	def __init__(self, input_size, output_size, hidden_size, norm):
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

		self.batch_size = 32
		self.dropout = 0.7

		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size + 1])
		self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_size])

		l0 = TF_PFNN_Layer((hidden_size, input_size), elu_operator = tf.nn.elu)
		l1 = TF_PFNN_Layer((hidden_size, hidden_size), elu_operator =tf.nn.elu)
		l2 = TF_PFNN_Layer((output_size, hidden_size))

		self.add_layer(l0)
		self.add_layer(l1)
		self.add_layer(l2)

		self.train_step = None
		self.cost_function = None

	
	def build_tf_graph(self, params):
		"""
		Builds the network graph for tensorflow. This should not be called directly, but is used in the training function. 
		
		Arguments:
			params {list} -- Unused.
		"""
		p = self.x[:, -1]
		net_in = self.x[:, :-1]
		dropout_prob = tf.constant(self.dropout, tf.float32)

		params = [net_in, p]
		print("input shape: ", net_in.shape)
		params = super().build_tf_graph(params)
		output = params[0]
		print("output shape: ", output.shape)

		tf.global_variables_initializer()

		self.cost_function = tf.reduce_mean((self.y - output) ** 2) + 0.01 * (
							tf.reduce_mean(tf.abs(self.layers[0].weight)) + 
							tf.reduce_mean(tf.abs(self.layers[1].weight)) + 
							tf.reduce_mean(tf.abs(self.layers[2].weight))
							)
		
		opt = tf.train.AdamOptimizer(learning_rate = 0.0001)
		var_list = []
		for l in self.layers:
			var_list.append(l.weight)
			var_list.append(l.bias)
		self.train_step = opt.minimize(self.cost_function, var_list=var_list)

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
				acc = np.array([])

				batchind = np.arange(len(input) // self.batch_size)  # start batches from these indices.
				np.random.shuffle(batchind)
				
				step_counter = 0
				for i in range(0, len(input), self.batch_size):
					if (i + self.batch_size) >= len(input):
						break
					x = input[i:(i + self.batch_size), :]
					y = output[i:i + self.batch_size]
					#y = y_data.reshape(self.batch_size, self.output_shape)
					_, loss_value = sess.run([self.train_step, self.cost_function], feed_dict={self.x: x, self.y: y})
					
					step_counter += 1
					if i > 0 and (i % 20) == 0:
						sys.stdout.write(
							'\r[Epoch %3i] % 3.1f%% est. remaining time: %i min, %.5f acc' % (
						e, 100 * i / len(input), ((datetime.datetime.now() - start_time).total_seconds() / i * (len(input) - i)) / 60, loss_value))
						sys.stdout.flush()
				
				# save epoch: 
				#os.makedirs(target_path + "/epoch_%d"%e, exist_ok=True)
				self.store(target_path + "/epoch_%d.json"%e)


	def from_file(dataset, target_path, epochs):
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
			Xstd[Xstd == 0] = 1.0
			Ystd[Ystd == 0] = 1.0
			X = (X - Xmean) / Xstd
			Y = (Y - Ymean) / Ystd
			
			#np.savez("%s/Ymean"%target_path, Ymean)
			#np.savez("%s/Xmean"%target_path, Xmean)
			#np.savez("%s/Ystd"%target_path, Ystd)
			#np.savez("%s/Xstd"%target_path, Xstd)
			norm = {"Xmean": Xmean.tolist(), 
					"Ymean":Ymean.tolist(), 
					"Xstd":Xstd.tolist(), 
					"Ystd":Ystd.tolist()}

			input_dim = X.shape[1]
			output_dim = Y.shape[1]

			print("in-out: ", input_dim, output_dim)
			X = np.concatenate([X, P[:, np.newaxis]], axis=-1)


			pfnn = PFNN(input_dim, output_dim, 512, norm)
			pfnn.train(X, Y, epochs, target_path)