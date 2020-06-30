import json
import numpy as np

from .layers.layers import Layer

class FCNetwork:
	"""
	FCNetwork is an abstract implementation of a fully connected neural network. 
	This class should not be directly used, but inherited by other fully connected networks. 

	Implemented member functions: 
		* add_layer: adds a layer to the network
		* build_tf_graph: iterates all layers to build the tf_graph
		* forward_pass normalizes input data and performs the forward pass of all layers
		* store: stores the networks as well as the data of all layers in a json file. 
	
	Not implemented member functions: 
		* load: loading the network from a json file -> requires knowledge about the layer classes
		* train: just used in tf implementation
		* all functions can be overwritten by child classes. 

	@author: Janis
	"""

	def __init__(self, input_size, output_size, hidden_size, norm):
		"""

		FCNetwork is an abstract implementation of a fully connected neural network. 
		This class should not be directly used, but inherited by other fully connected networks. 

		Implemented member functions: 
			* add_layer: adds a layer to the network
			* build_tf_graph: iterates all layers to build the tf_graph
			* forward_pass normalizes input data and performs the forward pass of all layers
			* store: stores the networks as well as the data of all layers in a json file. 
		
		Not implemented member functions: 
			* load: loading the network from a json file -> requires knowledge about the layer classes
			* train: just used in tf implementation
			* all functions can be overwritten by child classes. 

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

	def add_layer(self, layer : Layer):
		"""

		Adds a layer to the network. 
		
		Arguments:
			layer {Layer} -- New layer of Layer class (any realization possible)
		"""
		self.layers.append(layer)

	def build_tf_graph(self, params):
		"""

		Iteratively calls build_tf_graph(params) on all layers. 
		
		Arguments:
			params {list} -- initial list of parameters
		
		Returns:
			params {list} -- returns the updated list of parameters (updated by layers)
		"""
		for l in range(len(self.layers)):
			params = self.layers[l].build_tf_graph(params)
		return params

	def forward_pass(self, params):
		"""

		Performs the forward pass by iteratively calling the layer.forward_pass(params). 

		If the input string is of length 0, Xmean is used. 
		
		Arguments:
			params {list} -- list of parameters to be passed and updated by network layers. 
					params[0] is considered to contain the network input-string, the remaining parameters are unspecified. 
		
		Returns:
			params{list} -- updated parameter list. 
		"""
		# if len(params[0]) == 0:
		# 	params[0] = np.array(self.norm["Xmean"])
		# params[0] = (params[0] - self.norm["Xmean"]) / self.norm["Xstd"]
		# for l in self.layers:
		# 	params = l.forward_pass(params)
		#
		# params[0] = (params[0] * self.norm["Ystd"]) + self.norm["Ymean"]
		# return params

		if len(params) == 0:
			params = np.array(self.norm["Xmean"])

		params = (params - self.norm["Xmean"]) / self.norm["Xstd"]
		for l in self.layers:
			params = l.forward_pass(params)

		params = (params * self.norm["Ystd"]) + self.norm["Ymean"]
		params = np.array(params)
		return params


	def store(self, target_file):
		"""
		Stores the whole network in a json file. 
		
		Arguments:
			target_file {string} -- path to the target *.json file. 
		"""
		store = {"nlayers":(len(self.layers)),
				"input_size":(self.input_size),
				"output_size":(self.output_size),
				"hidden_size":(self.hidden_size),
				"norm":self.norm}

		for l in range(len(self.layers)):
			store["layer_%d"%l] = self.layers[l].store()

		with open(target_file, "w") as f:
			json.dump(store, f)

	@staticmethod
	def load(target_file):
		"""

		Loads the network specification from a *.json file. 

		Not implemented in abstract class!
		
		Arguments:
			target_file {string} -- path to *.json file. 
		"""
		pass

	def train(self, X, Y, epochs, target_path):
		"""
		Not implemented in abstract class!
		
		Arguments:
			X {np.array} -- Numpy array containing the input data (n_frames, x_dim)
			Y {np.array} -- Numpy array containing the output data (n_frames, y_dim)
			epochs {int} -- Training duration in epochs
			target_path {string} -- Path to a folder, where the network iterations should be stored. 
		"""
		pass