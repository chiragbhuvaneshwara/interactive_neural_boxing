from .layers import FCLayer, Interpolating_Layer, cubic
import numpy as np


class NP_FCLayer(FCLayer):

	def __init__(self, dshape, weight, bias, elu_operator = None):
		super().__init__(dshape, weight, bias, elu_operator)


	def forward_pass(self, params):
		input_string = params
		print("params: ", input_string.shape, self.weight.shape, self.bias.shape)
		d = np.matmul(self.weight, input_string) + self.bias
		if self.elu_operator is not None:
			d = self.elu_operator(d)
		return d


class NP_PFNN_Layer(Interpolating_Layer):
	
	def __init__(self, dshape, weight, bias, elu_operator = None):

		def interpolation_function(w, phase):
			pscale = 4 * phase
			pamount = pscale % 1.0
			p1 = int(pscale) % 4
			p0 = (p1 - 1) % 4
			p2 = (p1 + 1) % 4
			p3 = (p1 + 2) % 4
			
			dw = cubic(w[p0], w[p1], w[p2], w[p3], pamount)
			return dw

		super().__init__(dshape, weight, bias, elu_operator, NP_FCLayer, interpolation_function)
		
	def load(params):
		store = params[0]
		dshape = np.array(store["dshape"])
		weight = np.array(store["weight"])
		bias = np.array(store["bias"])
		elu = params[1]
		return NP_PFNN_Layer(dshape, weight, bias, elu)



