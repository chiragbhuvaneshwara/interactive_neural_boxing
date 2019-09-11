
"""
Variational network created for an experiment.
Basically the layers use random noise for the motion generation.
The network is stripped to contain only the necessary functions.
"""

import tensorflow as tf
import numpy as np
import sys, os, datetime, json

from .layers.layers_tf import TF_PFNN_Layer
from .pfnn_tf import PFNN


class pfnn_layer_with_random_noise(TF_PFNN_Layer):
	"""
	Builds on the Phase functional layer, however it also contains some random noise
		added to the output of each layer

	The following methods are used directly/indirectly from TF_PFNN_Layer:
		1. __init__
		3. set_random_state
		4. store()

	"""
	def __init__(self, dshape, weight = [], bias = [], elu_operator = None, name = ""):
		super().__init__(dshape, weight, bias, elu_operator, name)


	def set_random_state(self, shape=(1,), mean=0.0, std_dev=1.0):
		self.random_noise_to_add = tf.random.normal(shape, mean=mean, stddev=std_dev)

	def build_tf_graph(self, params):
		params = super().build_tf_graph(params)
		params[0] = tf.add(params[0], self.random_noise_to_add)
		return params

	def forward_pass(self, params):
		super().forward_pass(params)
		params[0] = tf.add(params[0], self.random_noise_to_add)
		return params

	def load(params):
		"""
		This constant function loads the network from a map store. 
		the params[0] field should contain a map containing:
			* dshape: Shape of the interpolated network
			* weight: weights
			* bias: biases
		
		params[1] contains the elu operator

		This function is not yet fully implemented!
		
		Arguments:
			params {list} -- parameters
		
		Returns:
			TF_PFNN_Layer -- generated layer. 
		"""
		# store = params[0]
		# dshape = np.frombuffer(store["dshape"], dtype=np.float32)
		# weight = np.frombuffer(store["weight"], dtype=np.float32)
		# bias = np.frombuffer(store["bias"], dtype=np.float32)
		# elu = params[1]
		store = params[0]
		dshape = np.array(store["dshape"], dtype=np.float32)
		weight = np.array(store["weight"], dtype=np.float32)
		bias = np.array(store["bias"], dtype=np.float32)
		elu = params[1]

		return pfnn_layer_with_random_noise(dshape, weight, bias, elu)



class pfnn_random_layers(PFNN):
	"""

	Implementation of variational interpolating neural networks using tensorflow backend. 

	This class is a realization of FCNetwork. 

	Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

	@author: Janis
	"""

	def __init__(self, input_size, output_size, hidden_size, norm, batch_size = 32, layers = [], dropout = 0.7, replace_layers = [], draw_each_layer = False, random_number_dim = -1):
		#input_size, output_size, hidden_size, norm, batch_size = 32, layers = [], dropout = 0.7):
		"""

		Implementation of variational interpolating neural networks using tensorflow backend. 

		This class is a realization of FCNetwork. 

		Please consider using the from_file function, which loads a numpy datastructure containing training pairs and constructs and trains a network. 

		Arguments:
			input_size {int} -- size of the input vector
			output_size {int} -- size of the output vector
			hidden_size {int} -- size of the hidden layers
			norm {map} -- map, containing the normalization information: Xmean, Ymean, Xstd, Ystd. 
		"""

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.norm = norm
		
		self.layers = []

		self.draw_each_layer, self.random_number_dim = draw_each_layer, random_number_dim
		if self.random_number_dim < 0:
			self.random_number_dim = hidden_size

		l0, l1, l2 = layers[0], layers[1], layers[2]

		self.batch_size = batch_size
		self.dropout = dropout

		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name="InputString")
		self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_size], name="OutputString")
		self.p = tf.placeholder(tf.float32, shape=[self.batch_size], name = "Phase")	

		self.counter = tf.placeholder(tf.int32, name="step_counter")

		self.add_layer(l0)
		self.add_layer(l1)
		self.add_layer(l2)
		self.replace_layers = replace_layers

		self.train_step = None
		self.cost_function = None
		self.first_decay_steps = 0


	def load(target_file, random_noise_dim, layers_to_add_random_noise):
		"""
			@param : random_layers - python list containing the layers to keep as random noise
		"""
		with open(target_file, "r") as f:
			store = json.load(f)
			input_size = store["input_size"]
			output_size = store["output_size"]
			hidden_size = store["hidden_size"]
			norm = store["norm"]			
			nlayers = store["nlayers"]
			if not nlayers == 3:
				print("ERROR: layers not matching. Stored: " + nlayers)
				return

			replace_layers = []

			if 0 in layers_to_add_random_noise:
				l0 = pfnn_layer_with_random_noise.load([store["layer_0"], tf.nn.elu])
				if random_noise_dim < 0:
					l0.set_random_state()
				else:
					size = (1, hidden_size)
					l0.set_random_state(size)
			else:
				l0 = TF_PFNN_Layer.load([store["layer_0"], tf.nn.elu])

			if 1 in layers_to_add_random_noise:
				l1 = pfnn_layer_with_random_noise.load([store["layer_1"], tf.nn.elu])
				if random_noise_dim < 0:
					l1.set_random_state()
				else:
					size = (1, hidden_size)
					l1.set_random_state(size)
			else:
				l1 = TF_PFNN_Layer.load([store["layer_1"], tf.nn.elu])			

			if 2 in layers_to_add_random_noise:
				l2 = pfnn_layer_with_random_noise.load([store["layer_2"], tf.nn.elu])
				if random_noise_dim < 0:
					l2.set_random_state()
				else:
					size = (1, output_size)
					l2.set_random_state(size)
			else:
				l2 = TF_PFNN_Layer.load([store["layer_2"], tf.nn.elu])

			layers = [l0, l1, l2]

			pfnn_random = pfnn_random_layers(input_size, output_size, hidden_size, norm, batch_size = 1, layers = layers, replace_layers=replace_layers, dropout=1)
			return pfnn_random

	
	def build_tf_graph(self, params):
		"""
		Builds the network graph for tensorflow. This should not be called directly, but is used in the training function. 
		
		Arguments:
			params {list} -- Unused.
		"""
		print(self.random_number_dim)
		if self.draw_each_layer:
			for l in self.replace_layers:
				z = tf.random.normal((self.batch_size, self.random_number_dim), 0.0, 1.0, dtype = tf.float32)
				self.layers[l].set_random_state(z)
		else:
			z = tf.random.normal((self.batch_size, self.random_number_dim), 0.0, 1.0, dtype = tf.float32)
			for l in self.replace_layers:
				self.layers[l].set_random_state(z)

		return super().build_tf_graph(params)


	def start_tf(self):
		self.build_tf_graph([])
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())


	def forward_pass(self, params):
		"""
		@param : params - Python list containing the data needed to run the network
			0 - input data
			1 - phase information
		@returns : out - output of the network, i.e. the prediction of next step of motion
			phase - the current phase
		"""
		# normalize: 
		params[0] = np.array(params[0])
		if len(params[0]) == 0:
			params[0] = np.array(self.norm["Xmean"])
		params[0] = (params[0] - self.norm["Xmean"]) / self.norm["Xstd"]

		# draw new random sample
		out = self.sess.run([self.network_output], feed_dict={self.x: [params[0]], self.p: [params[1]]})
		out = np.array(out[0][0])
		out = (out * self.norm["Ystd"]) + self.norm["Ymean"]

		return [out, params[1]]
