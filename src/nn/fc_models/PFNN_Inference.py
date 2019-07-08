import numpy as np
import math, time, os, sys
from helpers.helpers import *

class PFNN:

	def __init__(self, xdim, ydim, xmean, ymean, xstd, ystd, W0, W1, W2, b0, b1, b2, on_the_fly = False):
		self.xdim = xdim # 222 # 438 #450
		self.ydim = ydim  #217 # 559
		self.hdim = 512

		self.xmean, self.xstd= xmean, xstd
		self.ymean, self.ystd = ymean, ystd
		self.W0, self.W1, self.W2 = W0, W1, W2
		self.b0, self.b1, self.b2 = b0, b1, b2

		self.precomputed_weights = {}
		self.precomputed_bins = 50
		self.use_bins = True
		if not on_the_fly:
			self.precompute_bins()


	def precompute_bins(self):
		for pbin in range(0, self.precomputed_bins):
			p = pbin / self.precomputed_bins
			# the actual computation takes too long for practice.
			w0 = self.cubic_matrix(self.W0, p, self.xdim, self.hdim)
			w1 = self.cubic_matrix(self.W1, p, self.hdim, self.hdim)
			w2 = self.cubic_matrix(self.W2, p, self.hdim, self.ydim)

			b0 = self.cubic_bias(self.b0, p, self.hdim)
			b1 = self.cubic_bias(self.b1, p, self.hdim)
			b2 = self.cubic_bias(self.b2, p, self.ydim)

			self.precomputed_weights[pbin] = {}
			self.precomputed_weights[pbin]["w0"] = w0
			self.precomputed_weights[pbin]["w1"] = w1
			self.precomputed_weights[pbin]["w2"] = w2
			self.precomputed_weights[pbin]["b0"] = b0
			self.precomputed_weights[pbin]["b1"] = b1
			self.precomputed_weights[pbin]["b2"] = b2


	@classmethod
	def from_file(cls, path, mean_folder):

		def load_weights(name):
			return np.fromfile("%s/%s.bin" % (path, name), np.float32)

		# def load_weights(self, name, shape):
		# 	a = np.fromfile(name, dtype=np.float32)
		# 	a = a.reshape(shape)
		# 	return a
		#
		# def load_biases(self, name, shape):
		# 	a = np.fromfile(name, dtype=np.float32)
		# 	a = a.reshape(shape)
		# 	return a

		Xmean, Ymean = np.load("%s/Xmean.bin.npy"%mean_folder), np.load("%s/Ymean.bin.npy"%mean_folder)#np.fromfile("%s/Xmean.bin.npy"%mean_folder, np.float32), np.fromfile("%s/Ymean.bin.npy"%mean_folder, np.float32)
		if os.path.isfile("%s/Xstd.bin.npy"%mean_folder):
			Xstd, Ystd = np.load("%s/Xstd.bin.npy"%mean_folder), np.load("%s/Ystd.bin.npy"%mean_folder)#np.fromfile("%s/Xstd.bin.npy" % mean_folder, np.float32), np.fromfile("%s/Ystd.bin.npy" % mean_folder, np.float32)
		else:
			Xstd, Ystd = np.ones(len(Xmean)), np.ones(len(Ymean))
		Xmean, Xstd = np.reshape(Xmean, (len(Xmean), 1)), np.reshape(Xstd, (len(Xmean), 1))
		Ymean, Ystd = np.reshape(Ymean, (len(Ymean), 1)), np.reshape(Ystd, (len(Ymean), 1))
		# Xstd[-1] = 1.0
		# Ystd[-1] = 1.0

		W0 = np.reshape(load_weights("W0"), (4, len(Xmean), 512))
		W1 = np.reshape(load_weights("W1"), (4, 512, 512))
		W2 = np.reshape(load_weights("W2"), (4, 512, len(Ymean)))
		b0 = np.reshape(load_weights("b0"), (4, 512, 1))
		b1 = np.reshape(load_weights("b1"), (4, 512, 1))
		b2 = np.reshape(load_weights("b2"), (4, len(Ymean), 1))

		input_shape = len(W0) // (4 * 512)
		output_shape = len(b2) // 4
		return cls(len(Xmean), len(Ymean), Xmean, Ymean, Xstd, Ystd, W0, W1, W2, b0, b1, b2)
	def cubic_matrix(self, weights, phase, inshape, outshape):
		# this amount of work is done, to keep the Catmull-Rom Spline periodic.
		pscale = 4 * phase
		pamount = pscale % 1.0
		p1 = int(pscale) % 4
		p0 = (p1 - 1) % 4
		p2 = (p1 + 1) % 4
		p3 = (p1 + 2) % 4
		# call cubic interpolation based on control points
		Wi = np.reshape(cubic(weights[p0], weights[p1], weights[p2], weights[p3], pamount), (outshape, inshape))
		return Wi

	def cubic_bias(self, bias_w, phase, outshape):
		# can use the same helper function as with weights.
		return self.cubic_matrix(bias_w, phase, 1, outshape)


	def predict(self, input_string, p):
		# helper function to interpolate the matrix.

		# lazy initialization of phase bins.
		pbin = int(p * self.precomputed_bins)
		if pbin in self.precomputed_weights and self.use_bins:
			# speed up ~x3
			w0 = self.precomputed_weights[pbin]["w0"]
			w1 = self.precomputed_weights[pbin]["w1"]
			w2 = self.precomputed_weights[pbin]["w2"]
			b0 = self.precomputed_weights[pbin]["b0"]
			b1 = self.precomputed_weights[pbin]["b1"]
			b2 = self.precomputed_weights[pbin]["b2"]
		else:
			# the actual computation takes too long for practice.
			w0 = self.cubic_matrix(self.W0, p, self.xdim, self.hdim)
			w1 = self.cubic_matrix(self.W1, p, self.hdim, self.hdim)
			w2 = self.cubic_matrix(self.W2, p, self.hdim, self.ydim)

			b0 = self.cubic_bias(self.b0, p, self.hdim)
			b1 = self.cubic_bias(self.b1, p, self.hdim)
			b2 = self.cubic_bias(self.b2, p, self.ydim)

			self.precomputed_weights[pbin] = {}
			self.precomputed_weights[pbin]["w0"] = w0
			self.precomputed_weights[pbin]["w1"] = w1
			self.precomputed_weights[pbin]["w2"] = w2
			self.precomputed_weights[pbin]["b0"] = b0
			self.precomputed_weights[pbin]["b1"] = b1
			self.precomputed_weights[pbin]["b2"] = b2

		def elu(a):
			return np.maximum(a, 0) + np.exp(np.minimum(a, 0)) - 1

		# forward evaluation
		input = np.array((input_string - self.xmean) / self.xstd)
		#pi = int(p * self.sample_points)
		H0 = elu(np.matmul(w0, input) + b0)
		H1 = elu(np.matmul(w1, H0) + b1)
		out = (np.matmul(w2, H1) + b2)

		out = (out * self.ystd) + self.ymean
		# ("Prediction needed: %f ms"%(time.time() - start)
		return out
