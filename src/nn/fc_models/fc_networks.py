import json
import numpy as np

class FCNetwork:

	def __init__(self, input_size, output_size, hidden_size, norm):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.norm = norm
		
		self.layers = []

	def add_layer(self, layer):
		self.layers.append(layer)

	def build_tf_graph(self, params):
		for l in range(len(self.layers)):
			params = self.layers[l].build_tf_graph(params)
		return params

	def forward_pass(self, params):
		if len(params[0]) == 0:
			params[0] = np.array(self.norm["Xmean"])
		params[0] = (params[0] - self.norm["Xmean"]) / self.norm["Xstd"]
		for l in self.layers:
			params = l.forward_pass(params)

		params[0] = (params[0] * self.norm["Ystd"]) + self.norm["Ymean"]
		return params


	def store(self, target_file):
		store = {"nlayers":(len(self.layers)),
				"input_size":(self.input_size),
				"output_size":(self.output_size),
				"hidden_size":(self.hidden_size), 
				"norm":self.norm}
		
		for l in range(len(self.layers)):
			store["layer_%d"%l] = self.layers[l].store()

		with open(target_file, "w") as f:
			json.dump(store, f)

	def load(target_file):
		pass
	
	def train(X, Y, epochs, target_path):
		pass