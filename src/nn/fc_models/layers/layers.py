
class Layer:
	"""
	This is an abstract class providing a neural network layer. 

	There are multiple realizations of this class. 

	@author: Janis

	"""

	def __init__(self):
		pass
	
	def build_tf_graph(self, params):
		pass

	def forward_pass(self, params):
		pass
	
	def store(self):
		pass
	
	def load(params):
		pass


class FCLayer(Layer):
	"""
	FCLayer is the base implementation of a fully connected layer. 
	Please remember, that this is an abstract class and the specific 
	implementation using tensorflow or numpy backend have to be used. 
	
	@author: Janis
	"""
	def __init__(self, dshape, weight, bias, elu_operator = None):
		"""
		FCLayer is the base implementation of a fully connected layer. 
		Please remember, that this is an abstract class and the specific 
		implementation using tensorflow or numpy backend have to be used. 
		
		"""
		self.dshape = dshape
		self.weight = weight
		self.bias = bias
		self.elu_operator = elu_operator


class Interpolating_Layer(Layer):
	"""
	An interpolating layer is generating specific networks using an interpolation function (e.g. phase function)
	and an interpolation factor (e.g. phase value)

	This class is an abstract implementation of a interpolating network. Please use the 
	specific realizations using tensorflow or numpy backend. 


	@Author: Janis
	"""
	
	def __init__(self, dshape, weight, bias, elu_operator, fclayer_class, interpolation_function):
		"""
		An interpolating layer is generating specific networks using an interpolation function (e.g. phase function)
		and an interpolation factor (e.g. phase value). It requires a layer class and an interpolation function. 

		This class is an abstract implementation of a interpolating network. Please use the 
		specific realizations using tensorflow or numpy backend. 

		Arguments:
			dshape {tuple / list} -- shape of the target layer size
			weight {np.array / tf.variable} -- weights (not interpolated), ideally: (n, output, input) 
												with n being the number of control points
			bias {np.array / tf.variable} -- bias (not interpolated), ideally (n, output)
			elu_operator {function} -- elu operator function to be applied. None if no elu operation is required. 
			fclayer_class {Layer Class} -- Layer class which is used to synthesize the interpolated layers. 
			interpolation_function {function} -- Interpolation function used to generate the layer weights and biases. 
		"""
		self.dshape = dshape
		self.weight = weight
		self.bias = bias
		self.elu_operator = elu_operator
		self.fclayer_class = fclayer_class
		self.interpolation_function = interpolation_function

		self.precomputed_layers = {}

	def __construct__layer(self, interpolation_factor, dynamic = True):
		"""

		This function generates network layers interpolated based on the current weights. 
		This class provides dynamic programming (if dynamic == True). 
		This is usefull for synthesis, as the interpolation steps can be reused. 
		
		Arguments:
			interpolation_factor {float} -- interpolation factor (e.g. phase)
		
		Keyword Arguments:
			dynamic {bool} -- usage of dynamic programming (default: {True})
		
		Returns:
			[Layer] -- synthesized layer of class self.fclayer_class (as defined by init method)
		"""
		if interpolation_factor in self.precomputed_layers and dynamic:
			return self.precomputed_layers[interpolation_factor]	
		
		wtmp = self.interpolation_function(self.weight, interpolation_factor)
		btmp = self.interpolation_function(self.bias, interpolation_factor)
		layer = self.fclayer_class(self.dshape, wtmp, btmp, self.elu_operator)
	
		if dynamic:
			self.precomputed_layers[interpolation_factor] = layer
		return layer

	def forward_pass(self, params):
		"""
		Performs interpolation and forward pass using the interpolated layer. 
		
		Arguments:
			params {list} -- List of parameters. params[0] contains the layer input, params[1] the interpolation factor. 
		
		Returns:
			params {list} -- Updated list of parameters (layer output at params[0])
		"""
		input_string = params[0]
		interpolation_factor = params[1]
		layer = self.__construct__layer(interpolation_factor)
		return [layer.forward_pass(params[0]), params[1]]

	def build_tf_graph(self, params):
		"""
		Interpolates the layer in a non dynamic way and calls the build_tf_graph on the generated layer. 

		You have to ensure, that the interpolation function generates a tensorflow graph as well!
		
		Arguments:
			params {list} -- List of parameters. params[0] contains the variable for layer input, 
							params[1] the placeholder for phase interpolation
							params[2] contains the dropout variable
		
		Returns:
			params{list} -- updated parameters. 
		"""
		input_string = params[0]
		interpolation_factor = params[1]
		layer = self.__construct__layer(interpolation_factor, dynamic=False)
		params[0] = layer.build_tf_graph(params)
		return params

	def store(self):
		"""
		Creates a store for this layer. 
		
		Returns:
			map -- "dshape": shape, "weight":weights, "bias": bias. 
		"""
		store = {"dshape":self.dshape, 
			"weight":self.weight,
			"bias":self.bias}
		return store

# class Variational_Layer(FCLayer):
# 	def __init__(self, original_layer):
# 		self.original_layer = original_layer
# 		self.random_state = [0.0] * self.original_layer.dshape[0]
# 		self.weight = self.original_layer.weight
# 		self.bias = self.original_layer.bias
# 		self.dshape = self.original_layer.dshape
# 		self.elu_operator = self.original_layer.elu_operator


# 	def set_random_state(self, z):
# 		self.random_state = z
# 		print("random state: ", self.random_state)

# 	def build_tf_graph(self, params):
# 		pass

# 	def forward_pass(self, params):
# 		pass
	
# 	def store(self):
# 		self.original_layer.store()
	
# 	def load(original_layer, params):
# 		original_layer.load(params)

def cubic(y0, y1, y2, y3, mu):
	"""
	Cubic catmul-rom spline implementation
	"""
	return (
	(-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
	(y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
	(-0.5 * y0 + 0.5 * y2) * mu +
	(y1))

