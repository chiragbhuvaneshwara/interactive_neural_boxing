from .layers import FCLayer, Interpolating_Layer, cubic
import numpy as np


class NP_FCLayer(FCLayer):
	"""
	FCLayer is the base implementation of a fully connected layer. 
	This class provides the numpy implementation for FCLayer. 

	@author: Janis
	"""
	
	def __init__(self, dshape, weight, bias, elu_operator = None):
		"""
		FCLayer is the base implementation of a fully connected layer. 
		This class provides the numpy implementation for FCLayer. 

		Please consider the load function to load a json stored network configuration. 
		
		Arguments:
			dshape {list / tuple} -- shape of the layer
			weight {np.array} -- weights
			bias {np.array} -- bias
		
		Keyword Arguments:
			elu_operator {function} -- elu operation (default: {None})
		"""

		super().__init__(dshape, weight, bias, elu_operator)


	def forward_pass(self, params):
		"""
		Performs the forward pass of the network. 
		The input string is the params argument. 

		Pseudo-Code: 
			elu ( self.weight * params + self.bias)
		
		Arguments:
			params {np.array} -- layer input string
		
		Returns:
			np.array -- layer output string
		"""
		input_string = params
		#print("params: ", input_string.shape, self.weight.shape, self.bias.shape)
		d = np.matmul(self.weight, input_string) + self.bias
		if self.elu_operator is not None:
			d = self.elu_operator(d)
		return d


class NP_PFNN_Layer(Interpolating_Layer):
	"""
	This class provides the numpy implementation of a phase-functioned layer using a catmul-rom spline. 
	It is a realization of an interpolating layer for usage (no training)

	@author: Janis
	"""

	def __init__(self, dshape, weight, bias, elu_operator = None):
		"""
		This class provides the numpy implementation of a phase-functioned layer using a catmul-rom spline. 
		It is a realization of an interpolating layer for usage (no training)

		Arguments:
			dshape {list / tuple} -- shape of the (synthesized) layer
			weight {np.array} -- weights
			bias {np.array} -- bias

		Keyword Arguments:
			elu_operator {function} -- elu operation (default: {None})

		Returns:
			[type_in] -- [description]
		"""
		def interpolation_function(w, phase):
			# This function performs the splitting of variables to the four layers. 
			# this part makes the spline cyclic. 
			pscale = 4 * phase
			pamount = pscale % 1.0
			p1 = int(pscale) % 4
			p0 = (p1 - 1) % 4
			p2 = (p1 + 1) % 4
			p3 = (p1 + 2) % 4
			
			# cubic function can be directly utilized. 
			dw = cubic(w[p0], w[p1], w[p2], w[p3], pamount)
			return dw

		super().__init__(dshape, weight, bias, elu_operator, NP_FCLayer, interpolation_function)
	
	@staticmethod
	def load(params):
		"""
		This constant function loads the network from a map store. 
		the params[0] field should contain a map containing:
			* dshape: Shape of the interpolated network
			* weight: weights
			* bias: biases
		
		params[1] contains the elu operator
		
		Arguments:
			params {list} -- parameters
		
		Returns:
			NP_PFNN_Layer -- generated layer. 
		"""
		store = params[0]
		dshape = np.array(store["dshape"])
		weight = np.array(store["weight"])
		bias = np.array(store["bias"])
		elu = params[1]
		return NP_PFNN_Layer(dshape, weight, bias, elu)



