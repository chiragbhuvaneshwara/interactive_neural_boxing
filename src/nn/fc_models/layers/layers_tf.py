from .layers import FCLayer, Interpolating_Layer, cubic
import tensorflow as tf
import numpy as np



class TF_FCLayer(FCLayer):

	def __init__(self, dshape, weight, bias, elu_operator = None):
		super().__init__(dshape, weight, bias, elu_operator)

	def build_tf_graph(self, params):
		di = params[0]
		a = tf.reshape(tf.matmul(self.weight, di[:,:, tf.newaxis]), self.bias.shape) + self.bias
		if self.elu_operator is not None:
			a = self.elu_operator(a)
		return a


class TF_PFNN_Layer(Interpolating_Layer):

	def __init__(self, dshape, weight = [], bias = [], elu_operator = None):
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
			pscale = 4 * phase
			pamount = pscale % 1.0
			p1 = tf.cast(pscale, tf.int32) % 4
			p0 = (p1 - 1) % 4
			p2 = (p1 + 1) % 4
			p3 = (p1 + 2) % 4

			y0 = tf.nn.embedding_lookup(w, p0)
			y1 = tf.nn.embedding_lookup(w, p1)
			y2 = tf.nn.embedding_lookup(w, p2)
			y3 = tf.nn.embedding_lookup(w, p3)

			if len(y0.shape) == 3:
				pamount = pamount[:, tf.newaxis, tf.newaxis]
			if len(y0.shape) == 2:
				pamount = pamount[:, tf.newaxis]
			wi = cubic(y0, y1, y2, y3, pamount)
			
			return wi
		super().__init__(dshape, weight, bias, elu_operator, TF_FCLayer, interpolation_function)

	def load(params):
		store = params[0]
		dshape = np.frombuffer(store["dshape"], dtype=np.float32)
		weight = np.frombuffer(store["weight"], dtype=np.float32)
		bias = np.frombuffer(store["bias"], dtype=np.float32)
		elu = params[1]
		return TF_PFNN_Layer(dshape, weight, bias, elu)

	def store(self):
		store = {"dshape":np.array(self.dshape).tolist(), 
		"weight":self.sess.run(self.weight).astype(np.float32).tolist(),
		"bias":self.sess.run(self.bias).astype(np.float32).tolist()}
		return store

