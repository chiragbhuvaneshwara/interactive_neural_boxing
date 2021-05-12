This folder contains implementations of fully-connected feedfoward network models, similar to PFNN. 
The following architecture is proposed, to enable easier implementation and extension of network architectures. 


# Networks
*FCNetwork*: An abstract base class for fully connected networks. It contains layers and provides interfaces for forward_passes, building of tf_graphs, loading & storing of networks, training, etc. It has to be implemented to function correctly. However, controllers are utilizing the abstract interface to execute any arbitrary architecture trained with the same network model. 

Implementations:
	* pfnn_np / pfnn_tf: PFNN implementation with tensorflow (inference and training) and numpy backend (only inference)
	* vinn_tf : VINN implementation with tensorflow backend (inference and training), extends pfnn_tf

# Layers
*Layer*: Layer provides an abstract interface to describe network layers. An implemented layer should have the ability to generate the tf graph on its own and provides interfaces for forward-pass, loading and storing. There are different abstract extensions to the layer. 
	* FCLayer: Abstract class of a fully connected layer. Numerically it represents: elu (weights * input + bias)
	* Interpolating_Layer: Abstract class of an interpolating layer. An interpolating layer contains an "interpolating function" and generates fully connected layers based on an "interpolation factor". 

Implementations: 
	* TF_FCLayer: implementation of a simple tensorflow fully connected layer. Tensorflow variables have to be defined outside and should be added as parameters. 
	* TF_PFNN_Layer: implementation of a PFNN layer as an interpolating layer. Pre-calculated weights can be defined, but this one is generating tf.Variables. 
	* TF_Variational_Layer: extension of TF_PFNN_Layer learning the variance in hidden space. 

	* NP_FCLayer: similar to TF_FCLayer with numpy backend (only inference)
	* NP_PFNN_Layer: similar to TF_PFNN_Layer with numpy backend (only inference)