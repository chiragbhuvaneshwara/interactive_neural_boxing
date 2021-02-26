"""author: Janis Sprenger """
import numpy as np
import math, time

from ..nn.fc_models.fc_networks import FCNetwork
from .controller import Controller
from .trajectory import Trajectory
from .character import Character

from .. import utils

DEBUG = False
DEBUG_TIMING = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)

	

class DirectionalController(Controller):
	"""
	This is a controller for directional input. 
	
	Returns:
		[type_in] -- [description]
	"""
	def __init__(self, network : FCNetwork, config_store):#xdim, ydim, endJoints = 5, numJoints = 21):
		self.network = network

		self.initial = True
		self.xdim = network.input_size #xdim # 222 # 438 #450
		self.ydim = network.output_size #ydim #217 # 559
		#self.hdim = 512

		self.precomputed_weights = {}
		self.precomputed_bins = 50

		# self.endJoints = 5
		# self.joints = 21 + self.endJoints # 59
		self.endJoints = config_store["endJoints"]
		self.n_joints = config_store["numJoints"] + self.endJoints
		self.n_gaits = config_store["n_gaits"]
		self.use_rotations = config_store["use_rotations"]
		self.use_foot_contacts = config_store["use_footcontacts"]
		self.zero_posture = config_store["zero_posture"]

		input_data = np.array([0.0] * self.xdim)
		out_data = np.array([0.0] * self.ydim)
		self.input = PFNNInput(input_data, self.n_joints, self.n_gaits, self.endJoints, self.use_foot_contacts)
		self.output = PFNNOutput(out_data, self.n_joints, self.endJoints, self.use_foot_contacts)

		self.lastphase = 0

		self.target_vel = np.array((0.0, 0.0, 0.0))
		self.target_dir = np.array((0.0, 0.0, 0.0))

		self.traj = Trajectory()
		self.char = Character(config_store)
		self.config_store = config_store
		self.__initialize()

	def pre_render(self, direction, phase):
		"""
		This function is called before rendering. It prepares character and trajectory to be rendered. 
		
		Arguments:
			direction {np.array(3)} -- user input direction
			phase {float} -- walking phase
		
		Returns:
			float -- new phase
		"""
		
		# direction = normalize(direction)
		if DEBUG:
			print("\n\n############## PRE RENDER ###########")
			print("input dir: ", direction, "")
		if DEBUG_TIMING:
			start_time = time.time()

		# 1. Update target direction and velocity based on input
		# this currently does not model sideways stepping, as there is no camera direction involved. 
		target_vel_speed = 2.5 * np.linalg.norm(direction)   # target velocity factor, has to be adapted to dataset!												
		self.target_vel = utils.glm_mix(self.target_vel, target_vel_speed * direction, 0.9)  # 3d velocity mixed with old velocity

		target_vel_dir = self.target_dir if utils.euclidian_length(self.target_vel)  \
			< 1e-05 else utils.normalize(self.target_vel)  # get target direction, old direction if speed is too low
		
		self.target_dir = utils.mix_directions(self.target_dir, target_vel_dir, 0.9)  # mix with old target dir.

		if DEBUG:
			print("updated target_dir: ", self.target_vel, self.target_dir)

		# 2. update trajectory
		#self.update_target_dir_simple(direction)
		self.traj.compute_gait_vector(self.target_vel, self.char.running)
		self.traj.compute_future_trajectory(self.target_dir, self.target_vel)
		
		# 3. set trajectory input
		# set input
		input_pos, input_dir, traj_gait = self.traj.get_input(self.char.root_position, self.char.root_rotation, self.n_gaits)
		self.input.setTrajPos(input_pos)
		self.input.setTrajDir(input_dir)
		self.input.setTrajGait(traj_gait)

		# 4. prepare & set joint input
		# previous root transform to acurately map joint locations, velocities and rotations
		prev_root_pos, prev_root_rot = self.traj.getPreviousPosRot()
		joint_pos, joint_vel = self.char.getLocalJointPosVel(prev_root_pos, prev_root_rot)

		self.input.setJointPos(joint_pos)
		self.input.setJointVel(joint_vel)
		if self.use_rotations:
			self.input.setJointTwist(self.output.getRotations())  # joint twist is in local space anyways.
		
		# 5. predict
		if DEBUG_TIMING:
			pre_predict = time.time()
		[self.output.data, phase] = self.network.forward_pass([self.input.data, round(phase,2)])
		self.lastphase = phase
		if DEBUG_TIMING:
			post_predict = time.time()

		# 6. process prediction 
		# compute root transform
		self.char.root_position, self.char.root_rotation = self.traj.getWorldPosRot()

		if self.use_rotations:
			joint_rotations = self.output.getRotations() # twist rotations
		else:
			joint_rotations = []

		joint_positions = self.output.getJointPos()
		joint_velocities = self.output.getJointVel()
		foot_contacts = self.output.getFootContacts()

		foot_contacts[foot_contacts < 0.3] = 0
		foot_drifting = self.char.compute_foot_sliding(joint_positions, joint_velocities, foot_contacts)
		self.traj.foot_drifting = foot_drifting

		# 7. set new character pose

		self.char.set_pose(joint_positions, joint_velocities, joint_rotations)


		if DEBUG_TIMING:
			print("prerender: %f, from this predict: %f"%(time.time() - start_time, post_predict - pre_predict))
		return phase 

	def post_render(self):
		"""
		This function has to be called after rendering to prepare the next frame. 
		
		Returns:
			float -- changed phase depending on user output. 
		"""
		if DEBUG:
			print("\n\n############## POST RENDER ###########")
		if DEBUG_TIMING:
			start_time = time.time()
		stand_amount = self.traj.step_forward(self.output.getRotVel())
		
		# 1. update and smooth trajectory
		self.traj.update_from_predict(self.output.getNextTraj())
		if DEBUG:
			print("phase computation: ", stand_amount, self.output.getdDPhase(), self.lastphase, "")


		# 2. update phase
		self.lastphase = (self.lastphase + (stand_amount) * self.output.getdDPhase()) % (1.0)
		if DEBUG_TIMING:
			print("post_predict: %f"%(time.time() - start_time))
		return self.lastphase

	def reset(self, start_location, start_orientation, start_direction):
		"""
		Resets the controller to start location, orientation and direction. 
		
		Arguments:
			start_location {[type_in]} -- [description]
			start_orientation {[type_in]} -- [description]
			start_direction {[type_in]} -- [description]
		"""
		self.char.reset(start_location, start_orientation)
		self.traj.reset(start_location, start_orientation, start_direction)


	def copy(self):
		"""
		Should copy the controler. At the moment, just creates a new, blank controler. 
		
		Returns:
			[type_in] -- [description]
		"""
		return Controller(self.network, self.config_store)

	def getPose(self):
		"""
		This function forwards the posture for the next frame. Possibly just a forward from the character class

		Returns:
			np.array(float, (n_joints, 3)) - joint positions

		"""

		#root_pos, root_rot = self.traj.getWorldPosRot()
		root_pos, root_rot = np.array(self.char.root_position), np.array(self.char.root_rotation)
		pos, vel = self.char.getLocalJointPosVel(root_pos, root_rot)
		return np.reshape(pos, (self.char.joints, 3))
	
	def getWorldPosRot(self):
		return np.array(self.char.root_position), np.array(self.char.root_rotation)


	def __initialize(self):
		#self.set_weights(n.W0, n.W1, n.W2, n.b0, n.b1, n.b2, n.xmean, n.xstd, n.ymean, n.ystd)
		#self.network = n
		
		#if self.initial or True:
		self.output.data = np.array(self.network.norm["Ymean"])# * self.network.norm["Ystd"] + self.network.norm["Ymean"]
		self.input.data = np.array(self.network.norm["Xmean"])#* self.network.norm["Xstd"] + self.network.norm["Xmean"]


		[od, phase] = self.network.forward_pass([self.input.data, 0.0])
		
		joint_positions = self.output.getJointPos()
		joint_velocities = self.output.getJointVel()
		
		for j in range(0, self.n_joints):
			pos = self.char.root_position + utils.rot_around_z_3d(np.array([joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]]).reshape(3, ), self.char.root_rotation)
			vel = utils.rot_around_z_3d(np.array([joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]]).reshape(3, ), self.char.root_rotation)
			self.char.joint_positions[j] = pos
			self.char.joint_velocities[j] = vel

		if self.use_rotations:
			joint_rotations = self.output.getRotations()
			for j in range(0, len(joint_rotations)):
				self.char.joint_rotations[j] = joint_rotations[j]
		
		self.post_render()

			#self.initial = False

	# def getEndJointRotations(self):
	# 	return self.out[self.out_end_joint_rot_base:self.out_end_joint_rot_base + self.endJoints * 3]		

	def __update_target_dir_simple(self, direction):
		# Compute target direction from
		target_vel_speed = 2.5 # 0.05												# target velocity factor, has to be adapted to dataset!
		self.target_vel = direction * target_vel_speed
		self.target_dir = utils.normalize(np.array(self.target_vel))




class PFNNInput(object):
	"""
	This class is managing the network input. It is depending on the network data model
	
	Arguments:
		object {[type_in]} -- [description]
	
	Returns:
		[type_in] -- [description]
	"""
	def __init__(self, data, joints, n_gaits, endJoints, use_foot_contacts):
		self.data = data
		self.joints = joints

		self.n_gaits = n_gaits
		self.in_traj_pos_base = 0
		self.in_traj_dir_base = self.in_traj_pos_base + 2 * 12
		self.in_traj_gait_base = self.in_traj_dir_base + 2 * 12
		self.in_joint_pos = self.in_traj_gait_base + self.n_gaits * 12
		self.in_joint_vel = self.in_joint_pos + 3 * self.joints
		self.in_joint_twist = self.in_joint_vel + 3 * self.joints
		self.endJoints = endJoints
		return

	def setJointTwist(self, twist):
		self.data[self.in_joint_twist:self.in_joint_twist + self.joints - self.endJoints] = np.reshape(twist, [
			self.joints - self.endJoints, ])

	def setTrajPos(self, pos):
		self.data[self.in_traj_pos_base:self.in_traj_pos_base + 2 * 12] = pos[:] #np.reshape(pos, [24])

	def getInputTrajPos(self):
		return self.data[self.in_traj_pos_base:self.in_traj_pos_base + 2 * 12]

	def setTrajDir(self, dirs):
		self.data[self.in_traj_dir_base:self.in_traj_dir_base + 2 * 12] = dirs[:] #np.reshape(dirs, [24])

	def setTrajGait(self, gaits):
		self.data[self.in_traj_gait_base:self.in_traj_gait_base + self.n_gaits * 12] = gaits[:] #np.reshape(gaits,[gaits.size])

	def setJointPos(self, pos):
		self.data[self.in_joint_pos:self.in_joint_pos + 3 * self.joints] = pos[:] #np.reshape(pos, [pos.size, 1])

	def setJointVel(self, vel):
		self.data[self.in_joint_vel:self.in_joint_vel + 3 * self.joints] = vel #np.reshape(vel, [vel.size, 1])

	def getLastTrajPos(self):
		return np.array([self.data[self.in_traj_pos_base + 1:self.in_traj_pos_base + 7],
						 self.data[self.in_traj_pos_base + 13:self.in_traj_pos_base + 13 + 6]])

	def getLastTrajDir(self):
		return np.array([self.data[self.in_traj_dir_base + 1:self.in_traj_dir_base + 7],
						 self.data[self.in_traj_dir_base + 13:self.in_traj_dir_base + 13 + 6]])

	def getLastTrajGait(self):
		return self.data[self.in_traj_gait_base + self.n_gaits:self.in_traj_gait_base + self.n_gaits * 7]


class PFNNOutput(object):
	"""
	This class is managing the network output. It is depending on the network data model. 
	
	Arguments:
		object {[type_in]} -- [description]
	
	Returns:
		[type_in] -- [description]
	"""
	def __init__(self, data, joints, endJoints, use_foot_contacts):
		self.data = data
		self.joints = joints
		self.out_root_base = 0
		self.out_dphase_base = self.out_root_base + 4
		self.out_contacts_base = self.out_dphase_base + 1
		if use_foot_contacts:
			self.out_next_traj_base = self.out_contacts_base + 4
		else:
			self.out_next_traj_base = self.out_contacts_base
		self.out_joint_pos_base = self.out_next_traj_base + 2 * 2 * 6
		self.out_joint_vel_base = self.out_joint_pos_base + self.joints * 3
		self.out_joint_rot_base = self.out_joint_vel_base + self.joints * 3
		# self.out_end_joint_rot_base = self.out_joint_rot_base + self.joints
		self.endJoints = endJoints
		return

	def getFootContacts(self):
		return np.array(self.data[self.out_contacts_base:self.out_contacts_base + 4])

	def getRotVel(self):
		#return self.data[self.out_root_base:self.out_root_base + 3]
		return np.array(self.data[self.out_root_base:self.out_root_base + 4])

	def getdDPhase(self):
		return np.array(self.data[self.out_dphase_base])

	def getNextTraj(self):
		return np.array(self.data[self.out_next_traj_base:self.out_next_traj_base + (2 * 2 * 6)])

	def getJointPos(self):
		return np.array(self.data[self.out_joint_pos_base:self.out_joint_pos_base + 3 * self.joints])

	def getJointVel(self):
		return np.array(self.data[self.out_joint_vel_base:self.out_joint_vel_base + 3 * self.joints])

	def getRotations(self):
		return np.array(self.data[self.out_joint_rot_base:self.out_joint_rot_base + 1 * (self.joints - self.endJoints)])

