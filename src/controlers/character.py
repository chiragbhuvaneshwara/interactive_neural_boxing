"""author: Janis Sprenger """
import numpy as np
import math, time

from ..nn.fc_models.fc_networks import FCNetwork

from .. import utils
DEBUG = False
DEBUG_TIMING = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)



class Character:
	"""
	Character class contains information about the simulated character. 
	
	Returns:
		[type] -- [description]
	"""
	def __init__(self, config_store):# endJoints = 5, numJoints = 21):
		self.endJoints = config_store["endJoints"]
		self.joints = config_store["numJoints"] + self.endJoints# 59

		# fields for joint positions and velocities in global space. 
		self.joint_positions = np.array([[0.0, 0.0, 0.0]] * self.joints)
		self.joint_velocities = np.array([[0.0, 0.0, 0.0]] * self.joints)
		self.last_joint_positions = np.array(self.joint_positions)

		self.foot_left = [4,5]
		self.foot_right = [9, 10]

		self.local_joint_positions = np.array(self.joint_positions)
		self.joint_rotations = np.array([0.0] * self.joints)
		self.root_rotation = 0.0
		self.running = 0.0
		self.root_position = np.array([0.0, 0.0, 0.0])

	def reset(self, root_position, start_orientation):
		self.root_position = root_position
		self.root_rotation = start_orientation
		self.running = 0.0

	def set_pose(self, joint_positions, joint_velocities, joint_rotations, foot_contacts = [0, 0, 0, 0]):
		"""
		Sets a new pose after prediction. 
		
		Arguments:
			joint_positions {np.array(njoints * 3)} -- predicted root-local joint positions
			joint_velocities {np.array(njoints * 3)} -- predicted root-local joint velocity
			joint_rotations {[type]} -- not utilized at the moment
		
		Keyword Arguments:
			foot_contacts {list} -- binary vector of foot-contacts  (default: {[0, 0, 0, 0]})
		"""
		self.joint_rotations = joint_rotations
		# build local transforms from prediction
		self.last_joint_positions = np.array(self.joint_positions)

		for j in range(0, self.joints):
			local_pos = np.array([joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]]).reshape(3, )
			pos = utils.rot_around_z_3d(local_pos, self.root_rotation) + self.root_position
			local_vel = np.array([joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]]).reshape(3, )
			vel = utils.rot_around_z_3d(local_vel,self.root_rotation)

			self.joint_positions[j] = utils.glm_mix(self.joint_positions[j] + vel, pos, 0.5)  # mix positions and velocities.
			#self.local_joint_positions[j] = utils.rot_around_z_3d(self.joint_positions[j] - self.root_position, self.root_rotation, inverse = True)
			self.joint_velocities[j] = vel
		# prediction is finished and post processed. Pose can be rendered!
		return
	
	def compute_foot_sliding(self, joint_positions, joint_velocities, foot_contacts = [0, 0, 0, 0]):
		def compute_foot_movement(j):
			local_pos = np.array([joint_positions[j * 3 + 0], joint_positions[j * 3 + 1], joint_positions[j * 3 + 2]]).reshape(3, )
			pos = utils.rot_around_z_3d(local_pos, self.root_rotation) + self.root_position
			local_vel = np.array([joint_velocities[j * 3 + 0], joint_velocities[j * 3 + 1], joint_velocities[j * 3 + 2]]).reshape(3, )
			vel = utils.rot_around_z_3d(local_vel,self.root_rotation)
			return self.joint_positions[j] - utils.glm_mix(self.joint_positions[j] + vel, pos, 0.5)
		
		global_foot_drift = np.array([0.0,0.0,0.0])
		if foot_contacts[0]:
			global_foot_drift += compute_foot_movement(self.foot_left[0])
		if foot_contacts[1]:
			global_foot_drift += compute_foot_movement(self.foot_left[1])
		if foot_contacts[2]:
			global_foot_drift += compute_foot_movement(self.foot_right[0])
		if foot_contacts[3]:
			global_foot_drift += compute_foot_movement(self.foot_right[1])

		if (np.sum(foot_contacts) > 0):
			global_foot_drift /= np.sum(foot_contacts)
			global_foot_drift[1] = 0.0
		#print("foot correction: ", foot_contacts, global_foot_drift)
		#return np.array([0.0, 0.0, 0.0])
		return global_foot_drift

	def getLocalJointPosVel(self, prev_root_pos, prev_root_rot):
		joint_pos = np.array([0.0] * (self.joints * 3))
		joint_vel = np.array([0.0] * (self.joints * 3))

		for i in range(0, self.joints):
			# get previous joint position
			
			#pos = utils.rot_around_z_3d(self.char.joint_positions[i] - prev_root_pos, prev_root_rot, inverse=True)  # self.char.joint_positions[i]#
			pos = utils.rot_around_z_3d(self.joint_positions[i] - prev_root_pos, -prev_root_rot)  # self.char.joint_positions[i]#
			joint_pos[i * 3 + 0] = pos[0]
			joint_pos[i * 3 + 1] = pos[1]
			joint_pos[i * 3 + 2] = pos[2]

			# get previous joint velocity
			#vel = utils.rot_around_z_3d(self.char.joint_velocities[i], prev_root_rot,inverse=True)  # self.char.joint_velocities[i] #
			vel = utils.rot_around_z_3d(self.joint_velocities[i], -prev_root_rot)  # self.char.joint_velocities[i] #
			joint_vel[i * 3 + 0] = vel[0]
			joint_vel[i * 3 + 1] = vel[1]
			joint_vel[i * 3 + 2] = vel[2]
		
		return (joint_pos, joint_vel)


