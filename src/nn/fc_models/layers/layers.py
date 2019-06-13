
def cubic(y0, y1, y2, y3, mu):
	return (
	(-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
	(y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
	(-0.5 * y0 + 0.5 * y2) * mu +
	(y1))

class Layer:
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
	def __init__(self, dshape, weight, bias, elu_operator = None):
		self.dshape = dshape
		self.weight = weight
		self.bias = bias
		self.elu_operator = elu_operator


class Interpolating_Layer(Layer):

	def __init__(self, dshape, weight, bias, elu_operator, fclayer_class, interpolation_function):
		self.dshape = dshape
		self.weight = weight
		self.bias = bias
		self.elu_operator = elu_operator
		self.fclayer_class = fclayer_class
		self.interpolation_function = interpolation_function

		self.precomputed_layers = {}

	def __construct__layer(self, interpolation_factor, dynamic = True):
		if interpolation_factor in self.precomputed_layers and dynamic:
			return self.precomputed_layers[interpolation_factor]	
		
		wtmp = self.interpolation_function(self.weight, interpolation_factor)
		btmp = self.interpolation_function(self.bias, interpolation_factor)
		layer = self.fclayer_class(self.dshape, wtmp, btmp, self.elu_operator)
	
		if dynamic:
			self.precomputed_layers[interpolation_factor] = layer
		return layer

	def forward_pass(self, params):
		input_string = params[0]
		interpolation_factor = params[1]
		layer = self.__construct__layer(interpolation_factor)
		return [layer.forward_pass(params[0]), params[1]]

	def build_tf_graph(self, params):
		interpolation_factor = params[0]
		layer = self.__construct__layer(interpolation_factor, dynamic=False)
		return [params[0], layer.build_tf_graph(params[1:])]

	def store(self):
		store = {"dshape":self.dshape, 
			"weight":self.weight,
			"bias":self.bias}
		return store


