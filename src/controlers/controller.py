"""author: Janis Sprenger """
import numpy as np
import math, time

from ..nn.fc_models.fc_networks import FCNetwork

from .. import utils

class Controller:
	"""
	This is the base class for a NN controller. At the moment, only a directional controller is implemented. 
	Controllers are usually data-model dependent. 

	Basic usage of the controller: 

	1. pre_render - with input. Prepare the next state, including prediction
	2. getPose - get the joint position in local space
	3. getWorldPosRot - get root position and rotation in global space
	4. postRender - clean-up / blending for preparation of next step. 

	"""
	
	def pre_render(self, control_input):
		"""
		This function is called before rendering. It prepares everything for the next frame, including network prediction. 

		
		Arguments:
			control_input: has to be changed to match the specfic controller
		"""

		pass

	def getPose(self):
		"""
		This function returns the root-local joint positions

		Returns:
			np.array(float, (n_joints, 3)) - joint positions

		"""
		pass

	def getWorldPosRot(self):
		"""
		This function returns the global root position and rotation around up axis

		Returns:
			(pos, rot) = (np.array(float, 3), float)
		"""

	def post_render(self):
		"""
		This function has to be called after rendering to prepare the next frame. 
		"""
		pass

	def reset(self, start_location, start_orientation, start_direction):
		"""
		This function resets the agent at a certain location with a specific orientation in world coordinates. 
		
		Arguments:
			start_location {[type]} -- [description]
			start_orientation {[type]} -- [description]
			start_direction {[type]} -- [description]
		"""
		pass

	def copy(self):
		"""
		Creates a deep copy of a controller to enable scientific experiments. 
		"""
		pass

	