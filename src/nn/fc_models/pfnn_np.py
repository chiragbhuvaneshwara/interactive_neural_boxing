from .layers.layers_np import NP_PFNN_Layer
from .fc_networks import FCNetwork
import json
import numpy as np

class PFNN(FCNetwork):

	def __init__(self, input_size, output_size, hidden_size, norm):
		super().__init__(input_size, output_size, hidden_size, norm)

		def elu(a):
			return np.maximum(a, 0) + np.exp(np.minimum(a, 0)) - 1

		l0 = NP_PFNN_Layer((hidden_size, input_size), np.zeros((4, hidden_size, input_size)), np.zeros((4, hidden_size)), elu)
		l1 = NP_PFNN_Layer((hidden_size, hidden_size), np.zeros((4, hidden_size, hidden_size)), np.zeros((4, hidden_size)), elu)
		l2 = NP_PFNN_Layer((output_size, hidden_size), np.zeros((4, output_size, hidden_size)), np.zeros((4, output_size)), None)


		self.add_layer(l0)
		self.add_layer(l1)
		self.add_layer(l2)

	def load(target_file):
		def elu(a):
			return np.maximum(a, 0) + np.exp(np.minimum(a, 0)) - 1

		with open(target_file, "r") as f:
			store = json.load(f)

			nlayers = store["nlayers"]
			input_size = store["input_size"]
			output_size = store["output_size"]
			hidden_size = store["hidden_size"]
			jsonnorm = store["norm"]
			norm = {"Xmean":np.array(jsonnorm["Xmean"]), 
					"Ymean":np.array(jsonnorm["Ymean"]), 
					"Xstd":np.array(jsonnorm["Xstd"]), 
					"Ystd":np.array(jsonnorm["Ystd"])}


			if not nlayers == 3:
				print("ERROR: layers not matching. Stored: " + nlayers)
				return
			
			l0 = NP_PFNN_Layer.load([store["layer_0"], elu])
			l1 = NP_PFNN_Layer.load([store["layer_1"], elu])
			l2 = NP_PFNN_Layer.load([store["layer_2"], None])

			network = PFNN(input_size, output_size, hidden_size, norm)
			network.layers[0] = l0
			network.layers[1] = l1
			network.layers[2] = l2

			return network



