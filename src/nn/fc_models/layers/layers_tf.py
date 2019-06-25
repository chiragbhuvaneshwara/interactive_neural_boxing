from .layers import FCLayer, Interpolating_Layer, cubic
import tensorflow as tf
import numpy as np



class TF_FCLayer(FCLayer):
	"""
	FCLayer is the base implementation of a fully connected layer. 
	This class provides the tensorflow implementation for FCLayer. 

	@author: Janis
	"""
	def __init__(self, dshape, weight, bias, elu_operator = None):
		"""
		FCLayer is the base implementation of a fully connected layer. 
		This class provides the tensorflow implementation for FCLayer. 
		
		Arguments:
			dshape {list / tuple} -- shape of the layer
			weight {tf.variable} -- tensorflow object to the weights
			bias {tf.variable} -- tensorflow object to the bias
		
		Keyword Arguments:
			elu_operator {function} -- elu operation (e.g. tf.nn.elu) (default: {None})
		"""
		super().__init__(dshape, weight, bias, elu_operator)

	def build_tf_graph(self, params):
		"""
		Builds the tensorflow graph for the layer:
		The input field is considered to be given by params. 

		Pseudo-Code: 
			elu ( weight * params + bias ) 
		
		Arguments:
			params {tf.variable} -- network input tensorflow object
		
		Returns:
			tf.variable -- network output tensorflow object
		"""
		di = params
		a = tf.reshape(tf.matmul(self.weight, di[:,:, tf.newaxis]), self.bias.shape) + self.bias
		if self.elu_operator is not None:
			a = self.elu_operator(a)
		return a


class TF_PFNN_Layer(Interpolating_Layer):
	"""
	This class provides the tensorflow implementation of a phase-functioned layer using a catmul-rom spline. 
	It is a realization of an interpolating layer. 

	@author: Janis
	"""


	def __init__(self, dshape, weight = [], bias = [], elu_operator = None):
		"""
		This class provides the tensorflow implementation of a phase-functioned layer using a catmul-rom spline. 
		It is a realization of an interpolating layer. 

		
		Arguments:
			dshape {list / tuple} -- shape of the (synthesized) layer
		
		Keyword Arguments:
			weight {list} -- optional weight for fine-tuning (not yet implemented) (default: {[]})
			bias {list} -- optional weight for fine-tuning (not yet implemented) (default: {[]})
			elu_operator {function} -- elu operation (e.g. tf.nn.elu) (default: {None})

		"""
		if len(weight) == 0:
			W_bound = 1 * np.sqrt(6. / np.prod(dshape[1]))
			weight = np.asarray(np.random.uniform(low = -W_bound, high=W_bound, size=(4, dshape[0], dshape[1])), dtype=np.float32)
		
		weight = tf.Variable(weight, True)
		self.sess = None
		
		if len(bias) == 0:
			bias = tf.constant(0.0, shape = (4, dshape[0]))
		bias = tf.Variable(bias, True)

		print("initialized new layer: ", weight.shape, bias.shape)
		

		def interpolation_function(w, phase):
			# This function performs the splitting of variables to the four layers. 
			# this part makes the spline cyclic. 
			pscale = 4 * phase
			pamount = pscale % 1.0
			p1 = tf.cast(pscale, tf.int32) % 4
			p0 = (p1 - 1) % 4
			p2 = (p1 + 1) % 4
			p3 = (p1 + 2) % 4

			# y0 = tf.nn.embedding_lookup(w, p0)
			# y1 = tf.nn.embedding_lookup(w, p1)
			# y2 = tf.nn.embedding_lookup(w, p2)
			# y3 = tf.nn.embedding_lookup(w, p3)
			y0 = tf.gather(w, p0)
			y1 = tf.gather(w, p1)
			y2 = tf.gather(w, p2)
			y3 = tf.gather(w, p3)

			# pamount has to be rescaled appropriately, depending on weights (3 dimensions) and biases (2 dimensions)
			if len(y0.shape) == 3:
				pamount = pamount[:, tf.newaxis, tf.newaxis]
			if len(y0.shape) == 2:
				pamount = pamount[:, tf.newaxis]

			# interpolation using a cubic-catmul rom spline. 
			wi = cubic(y0, y1, y2, y3, pamount)
			
			return wi
		super().__init__(dshape, weight, bias, elu_operator, TF_FCLayer, interpolation_function)

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
		store = params[0]
		dshape = np.frombuffer(store["dshape"], dtype=np.float32)
		weight = np.frombuffer(store["weight"], dtype=np.float32)
		bias = np.frombuffer(store["bias"], dtype=np.float32)
		elu = params[1]
		return TF_PFNN_Layer(dshape, weight, bias, elu)

	def store(self):
		"""
		Generates a json store for the network configuration. 
		
		Returns:
			map -- json store
		"""
		store = {"dshape":np.array(self.dshape).tolist(), 
		"weight":self.sess.run(self.weight).astype(np.float32).tolist(),
		"bias":self.sess.run(self.bias).astype(np.float32).tolist()}
		return store

