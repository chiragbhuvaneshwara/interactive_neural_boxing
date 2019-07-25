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

class Trajectory:
	"""
	This class contains data and functionality for local trajectory control. 
	120 frames surrounding the current frame are considered. 
	Trajectory positions and directions are predicted by the network and blended with the user input. 
	This class manages gait information as well (as it is linked to the trajectory points). 
	
	Returns:
		[type] -- [description]
	"""
	def __init__(self):
		self.n_frames = 12 * 10  # 120 fps
		self.median_idx = self.n_frames // 2
		self.traj_positions = np.array([[0.0, 0.0, 0.0]] * self.n_frames)
		self.traj_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames)
		self.traj_rotations = np.array([0.0] * self.n_frames)

		self.traj_gait_stand = np.array([1.0] * self.n_frames)
		self.traj_gait_walk = np.array([0.0] * self.n_frames)
		self.traj_gait_jog = np.array([0.0] * self.n_frames)
		self.traj_gait_back = np.array([0.0] * self.n_frames)
		self.foot_drifting = np.array([0.0, 0.0, 0.0])
		self.blend_bias = 2.0

	def reset(self, start_location = [0,0,0], start_orientation = [1,0,0,0], start_direction = [0,0,1]):
		"""
		Resets the trajectory information and thus the character to 0,0,0 pointing to 0,0,1. 
		
		Keyword Arguments:
			start_location {list} -- [description] (default: {[0,0,0]})
			start_orientation {list} -- [description] (default: {[1,0,0,0]})
			start_direction {list} -- [description] (default: {[0,0,1]})
		"""
		self.traj_positions = np.array([[0.0, 0.0, 0.0]] * self.n_frames)
		self.traj_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames)
		self.traj_rotations = np.array([0.0] * self.n_frames)
		for idx in range(self.median_idx+1):
			self.traj_positions[idx] = np.array(start_location)
			self.traj_rotations[self.median_idx] = np.array(start_orientation)
			self.traj_directions[self.median_idx] = np.array(start_direction)

		self.traj_gait_stand = np.array([0.0] * self.n_frames)
		for idx in range(self.median_idx + 1):
			self.traj_gait_stand[idx] = 1
		self.traj_gait_walk = np.array([0.0] * self.n_frames)
		self.traj_gait_jog = np.array([0.0] * self.n_frames)
		self.traj_gait_back = np.array([0.0] * self.n_frames)

	def get_input(self, root_position, root_rotation, n_gaits):
		"""
		This function computes the network input vector based on the current trajectory state for the new root_position and root_rotations. 
		
		Arguments:
			root_position {[type]} -- new root position
			root_rotation {[type]} -- new root rotation
			n_gaits {[type]} -- number of gaits
		Returns:
			np.array[12 * 2] -- trajectory positions
			np.array[12 * 2] -- trajectory directions
			np.array[12 * n_gaits] -- trajectory gaits
		"""
		w = self.n_frames // 10  # only every 10th frame => resulting in w = 12
		input_pos = np.array([0.0] * 2 * w)
		input_dir = np.array([0.0] * 2 * w)
		if DEBUG:
			print("")
			print("char movement: ", root_position, math.degrees(root_rotation),
				  self.traj_directions[self.median_idx], "")
			print("")

		#frames_tmp_past = []
		#frames_tmp_future = []
		#for f in range(0, self.median_idx, 10):
		#    frames_tmp_past.append(self.traj_positions[f])
		#for f in range(self.median_idx, self.n_frames, 10):
		#    frames_tmp_future.append(self.traj_positions[f])

		for i in range(0, self.n_frames, 10):
			# root relative positions and directions of trajectories
			pos = utils.rot_around_z_3d(self.traj_positions[i] - root_position, root_rotation,
								  inverse=True)
			dir = utils.rot_around_z_3d(self.traj_directions[i], root_rotation, inverse=True)
			input_pos[w * 0 + i // 10] = pos[0]
			input_pos[w * 1 + i // 10] = pos[2]
			input_dir[w * 0 + i // 10] = dir[0]
			input_dir[w * 1 + i // 10] = dir[2]

		# gait input
		traj_gait = np.array([0.0] * (w * n_gaits))
		for i in range(0, self.n_frames, 10):
			traj_gait[w * 0 + i // 10] = self.traj_gait_stand[i]
			traj_gait[w * 1 + i // 10] = self.traj_gait_walk[i]
			traj_gait[w * 2 + i // 10] = self.traj_gait_jog[i]

		if DEBUG:
			print("input dir: ")
			print("x pos: ", input_pos[0:12], "")
			print("y pos: ", input_pos[12:], "")
			print("x dir: ", input_dir[:12], "")
			print("y dir: ", input_dir[12:], "")

			print("\nworld coords: ")
			print(self.traj_positions.shape)
			print("x pos: ", self.traj_positions[::10,0])
			print("y pos: ", self.traj_positions[::10,2])
			print("x dir: ", self.traj_directions[::10,0])
			print("y dir: ", self.traj_directions[::10,2])
			

		return input_pos, input_dir, traj_gait

	def compute_future_trajectory(self, target_dir, target_vel):
		"""
		Performs blending of the future trajectory for the next target direction and velocity. 
		
		Arguments:
			target_dir {np.array(3)} -- Direction
			target_vel {np.array(3)} -- Velocity
		"""
		# computing future trajectory
		trajectory_positions_blend = np.array(self.traj_positions)
		total_directional_error = np.array([0.0,0.0,0.0])
		for i in range(len(trajectory_positions_blend) // 2 + 1, len(trajectory_positions_blend)):
			# adjust bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
			scale_pos = 1.0 - pow(1.0 - (i - len(trajectory_positions_blend) // 2) / (1.0 * (len(trajectory_positions_blend) // 2)), self.blend_bias)
			dir_error = ((np.linalg.norm(trajectory_positions_blend[i-1] - self.traj_positions[self.median_idx]) * utils.normalize(target_vel)) - (trajectory_positions_blend[i-1] - self.traj_positions[self.median_idx]))
			trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + \
											utils.glm_mix(self.traj_positions[i] - self.traj_positions[i - 1],
													target_vel, scale_pos)
			#dir_error = np.linalg.norm((trajectory_positions_blend[i] - trajectory_positions_blend[i-1])) * (utils.normalize((target_vel)) - utils.normalize(trajectory_positions_blend[i] - self.traj_positions[self.median_idx]))
			#trajectory_positions_blend[i] += dir_error * scale_pos


			# adjust scale bias 0.5 to fit to dataset and responsivity (larger value -> more responsive)
			scale_dir = scale_pos
			# scale_dir = 1.0 - pow(
			# 	1.0 - (i - len(trajectory_positions_blend) // 2) * 1.0 / (len(trajectory_positions_blend) // 2),
			# 	self.blend_bias)
			self.traj_directions[i] = utils.mix_directions(self.traj_directions[i], target_dir,
														  scale_dir)  # self.target_vel

			# copy gait values
			self.traj_gait_stand[i] = self.traj_gait_stand[self.median_idx]
			self.traj_gait_walk[i] = self.traj_gait_walk[self.median_idx]
			self.traj_gait_jog[i] = self.traj_gait_jog[self.median_idx]
		
		for i in range(len(trajectory_positions_blend) // 2 + 1, len(trajectory_positions_blend)):
			self.traj_positions[i] = trajectory_positions_blend[i]

		# compute trajectory rotations
		for i in range(0, self.n_frames):
			self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])

	def compute_gait_vector(self, target_vel, running):
		"""
		Computes the future gait vector for a specific target velocity and running gait label
		
		Arguments:
			target_vel {np.array(3)} -- Velocity
			running {int[0,1]} -- gait
		"""
		# setting up gait vector
		if utils.euclidian_length(target_vel) < 0.001:
			# standing
			stand_amount = 1.0 - max(0.0, min(np.linalg.norm(target_vel) / 0.1, 1.0))
			self.traj_gait_stand[self.median_idx] = 0.9 * self.traj_gait_stand[self.median_idx] + 0.1 * stand_amount
			self.traj_gait_walk[self.median_idx] = 0.9 * self.traj_gait_walk[self.median_idx] + 0.0
			self.traj_gait_jog[self.median_idx] = 0.9 * self.traj_gait_jog[self.median_idx] + 0.0
		elif running > 0.1:
			# running
			self.traj_gait_stand[self.median_idx] = 0.9 * self.traj_gait_stand[self.median_idx] + 0.0
			self.traj_gait_jog[self.median_idx] = 0.9 * self.traj_gait_jog[self.median_idx] + 0.1
			self.traj_gait_walk[self.median_idx] = 0.9 * self.traj_gait_walk[self.median_idx] + 0.0
		else:
			# walking
			self.traj_gait_stand[self.median_idx] = 0.9 * self.traj_gait_stand[self.median_idx] + 0.0
			self.traj_gait_jog[self.median_idx] = 0.9 * self.traj_gait_jog[self.median_idx] + 0.0
			self.traj_gait_walk[self.median_idx] = 0.9 * self.traj_gait_walk[self.median_idx] + 0.1
		# print("gaits: ", self.traj_gait_stand[self.median_idx], self.traj_gait_walk[self.median_idx], self.traj_gait_jog[self.median_idx])

	def step_forward(self, rot_vel):
		"""
		Performs a frame-step after rendering
		
		Arguments:
			rot_vel {np.array(4)} -- root velocity + new root direction
		
		Returns:
			[type] -- [description]
		"""
		# mix positions with velocity prediction
		for i in range(0, len(self.traj_positions) // 2):
			self.traj_positions[i] = np.array(self.traj_positions[i + 1])
			self.traj_directions[i] = np.array(self.traj_directions[i + 1])
			self.traj_rotations[i] = self.traj_rotations[i + 1]

			self.traj_gait_walk[i] = self.traj_gait_walk[i + 1]
			self.traj_gait_stand[i] = self.traj_gait_stand[i + 1]
			self.traj_gait_jog[i] = self.traj_gait_jog[i + 1]

		## current trajectory
		stand_amount = math.pow(1.0 - self.traj_gait_stand[self.median_idx], 0.25)  # influence of gait stand.

		trajectory_update = utils.rot_around_z_3d(np.array([rot_vel[0], 0.0, rot_vel[1]]),
											self.traj_rotations[self.median_idx])

		#rotational_vel = rot_vel[2]
		# if rot_vel[2] < 0.00001 and rot_vel[3] < 0.00001:
		# 	rotational_vel = 0
		# elif rot_vel[2] < 0.00001:
		# 	rotational_vel = 0
		# elif rot_vel[3] < 0.00001:
		# 	if rot_vel[2] > 0:
		# 		rotational_vel = math.pi
		# 	else:
		# 		rotational_vel = - math.pi
		# else:
		# 	rotational_vel = math.atan(rot_vel[2] / rot_vel[3]) 
			
		rotational_vel = math.atan2(rot_vel[2], rot_vel[3])

		if DEBUG:
			print("trajectory update: ", stand_amount, stand_amount * trajectory_update, -stand_amount * rotational_vel, "")

		self.traj_positions[self.median_idx] = self.traj_positions[self.median_idx] + stand_amount * trajectory_update + self.foot_drifting
		self.traj_directions[self.median_idx] = utils.rot_around_z_3d(self.traj_directions[self.median_idx], -stand_amount * rotational_vel)
		if DEBUG:
			print(" new root_direction: ", self.traj_directions[self.median_idx], utils.z_angle(self.traj_directions[self.median_idx]), "")
		self.traj_rotations[self.median_idx] = utils.z_angle(self.traj_directions[self.median_idx])
		return stand_amount

	def update_from_predict(self, prediction):
		"""
		Update trajectory from prediction. 
		Prediction is assumed to contain first trajectory positions of the next 6 trajectory points for x, then y and afterwards directions, first for x then y. 
		
		Arguments:
			prediction {np.array(4 * (6 * 2))} -- vector containing the network output regarding future trajectory positions and directions
		"""
		root_rotation = self.traj_rotations[self.median_idx]
		root_pos = self.traj_positions[self.median_idx]
		if DEBUG:
			print("       root movement", self.traj_positions[self.median_idx],
				  self.traj_directions[self.median_idx], root_rotation, "")

		# update future trajectory based on prediction. Future trajectory will solely depend on prediction and will be smoothed with the control signal by the next pre_render
		for i in range(self.median_idx + 1, self.n_frames):
			w = self.median_idx // 10
			m = ((i - self.n_frames / 2) / 10.0) % 1.0
			# The following is only relevant to smooth between 0 / 10 and 1 / 10 steps, if 120 points are used

			self.traj_positions[i][0] = (1 - m) * prediction[w * 0 + (i // 10 - w)] + m * prediction[w * 0 + (i // 10 - w) + (1 if i < 109 else 0)]
			self.traj_positions[i][2] = (1 - m) * prediction[w * 1 + (i // 10 - w)] + m * prediction[w * 1 + (i // 10 - w) + (1 if i < 109 else 0)]
			self.traj_positions[i] = utils.rot_around_z_3d(self.traj_positions[i], root_rotation) + root_pos

			self.traj_directions[i][0] = (1 - m) * prediction[w * 2 + (i // 10 - w)] + m * prediction[w * 2 + (i // 10 - w) + (1 if i < 109 else 0)]
			self.traj_directions[i][2] = (1 - m) * prediction[w * 3 + (i // 10 - w)] + m * prediction[w * 3 + (i // 10 - w) + (1 if i < 109 else 0)]
			self.traj_directions[i] = (utils.rot_around_z_3d(utils.normalize(self.traj_directions[i]), root_rotation))

			self.traj_rotations[i] = utils.z_angle(self.traj_directions[i])
	
	def correct_foot_sliding(self, foot_sliding):
		self.traj_positions[self.median_idx] += foot_sliding



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
		#self.joint_rotations = np.array([np.array([0.0, 0.0, 0.0, 1.0])] * self.joints)
		self.joint_rotations = np.array([0.0] * self.joints)
		#self.end_joint_rotations = np.array([[0.0, 0.0, 0.0, 0.1]] * self.endJoints)
		# print(self.joint_rotations)
		self.root_rotation = 0.0
		self.running = 0.0
		self.root_position = np.array([0.0, 0.0, 0.0])
		# self.joint_anim_global_xform = np.array([[[0.0] * 4] * 4] * self.joints)
		# self.joint_local_anim_xform = np.array([[[0.0] * 4] * 4] * self.joints)
		# self.joint_local_pos = np.array([np.array([0.0, 0.0, 0.0])] * self.joints)

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

	
class PFNNInput(object):
	"""
	This class is managing the network input. It is depending on the network data model
	
	Arguments:
		object {[type]} -- [description]
	
	Returns:
		[type] -- [description]
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
		object {[type]} -- [description]
	
	Returns:
		[type] -- [description]
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



class Controller:
	"""
	This class is the controler. 
	
	Returns:
		[type] -- [description]
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
		print("using footcontacts: ", self.use_foot_contacts)
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
		self.__update_target_dir(direction)
		#self.update_target_dir_simple(direction)
		self.traj.compute_gait_vector(self.target_vel, self.char.running)
		self.traj.compute_future_trajectory(self.target_dir, self.target_vel)
		
		# set input
		self.__set_trajectory()
		self.__set_previous_pose()

		if DEBUG_TIMING:
			pre_predict = time.time()
		[self.output.data, phase] = self.network.forward_pass([self.input.data, round(phase,2)])
		self.lastphase = phase
		if DEBUG_TIMING:
			post_predict = time.time()

		self.__get_root_transform()
		self.__get_new_pose()

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
		self.traj.update_from_predict(self.output.getNextTraj())
		if DEBUG:
			print("phase computation: ", stand_amount, self.output.getdDPhase(), self.lastphase, "")


		# update phase
		self.lastphase = (self.lastphase + (stand_amount) * self.output.getdDPhase()) % (1.0)
		if DEBUG_TIMING:
			print("post_predict: %f"%(time.time() - start_time))
		return self.lastphase

	def reset(self, start_location, start_orientation, start_direction):
		"""
		Resets the controller to start location, orientation and direction. 
		
		Arguments:
			start_location {[type]} -- [description]
			start_orientation {[type]} -- [description]
			start_direction {[type]} -- [description]
		"""
		self.char.reset(start_location, start_orientation)
		self.traj.reset(start_location, start_orientation, start_direction)


	def copy(self):
		"""
		Should copy the controler. At the moment, just creates a new, blank controler. 
		
		Returns:
			[type] -- [description]
		"""
		return Controller(self.network, self.config_store)

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

	def __update_target_dir(self, direction):
		# Compute target direction from
		#target_vel_speed = 2.5 * np.linalg.norm(direction)  # 0.05												# target velocity factor, has to be adapted to dataset!
		#direction = utils.rot_around_z_3d(direction, self.char.root_rotation, True)
		target_vel_speed = 2.5 * np.linalg.norm(direction)   # 0.05												# target velocity factor, has to be adapted to dataset!
		self.target_vel = utils.glm_mix(self.target_vel, target_vel_speed * direction,
							  0.9)  # 3d velocity mixed with old velocity

		target_vel_dir = self.target_dir if utils.euclidian_length(self.target_vel) \
											< 1e-05 else utils.normalize(
			self.target_vel)  # get target direction, old direction if speed is too low
		#target_vel_dir = utils.rot_around_z_3d(target_vel_dir, self.char.root_rotation, True)
		self.target_dir = utils.mix_directions(self.target_dir, target_vel_dir, 0.9)  # mix with old target dir.

		if DEBUG:
			print("updated target_dir: ", self.target_vel, self.target_dir)
		

	def __update_target_dir_simple(self, direction):
		# Compute target direction from
		target_vel_speed = 2.5 # 0.05												# target velocity factor, has to be adapted to dataset!
		self.target_vel = direction * target_vel_speed
		self.target_dir = utils.normalize(np.array(self.target_vel))

	def __get_root_transform(self):
		# compute root transform
		self.char.root_position = np.array(self.traj.traj_positions[self.traj.median_idx])
		self.char.root_position[1] = 0.0
		self.char.root_rotation = self.traj.traj_rotations[self.traj.median_idx]

	def __set_previous_pose(self):
		# previous root transform to acurately map joint locations, velocities and rotations
		prev_root_pos = self.traj.traj_positions[self.traj.median_idx - 1]
		prev_root_pos[1] = 0
		prev_root_rot = self.traj.traj_rotations[self.traj.median_idx - 1]

		joint_pos = np.array([0.0] * (self.n_joints * 3))
		joint_vel = np.array([0.0] * (self.n_joints * 3))
		for i in range(0, self.n_joints):
			# get previous joint position
			
			#pos = utils.rot_around_z_3d(self.char.joint_positions[i] - prev_root_pos, prev_root_rot, inverse=True)  # self.char.joint_positions[i]#
			pos = utils.rot_around_z_3d(self.char.joint_positions[i] - prev_root_pos, -prev_root_rot)  # self.char.joint_positions[i]#
			joint_pos[i * 3 + 0] = pos[0]
			joint_pos[i * 3 + 1] = pos[1]
			joint_pos[i * 3 + 2] = pos[2]

			# get previous joint velocity
			#vel = utils.rot_around_z_3d(self.char.joint_velocities[i], prev_root_rot,inverse=True)  # self.char.joint_velocities[i] #
			vel = utils.rot_around_z_3d(self.char.joint_velocities[i], -prev_root_rot)  # self.char.joint_velocities[i] #
			joint_vel[i * 3 + 0] = vel[0]
			joint_vel[i * 3 + 1] = vel[1]
			joint_vel[i * 3 + 2] = vel[2]

		self.input.setJointPos(joint_pos)
		self.input.setJointVel(joint_vel)
		if self.use_rotations:
			self.input.setJointTwist(self.output.getRotations())  # joint twist is in local space anyways.

	def __set_trajectory(self):
		input_pos, input_dir, traj_gait = self.traj.get_input(self.char.root_position, self.char.root_rotation, self.n_gaits)
		self.input.setTrajPos(input_pos)
		self.input.setTrajDir(input_dir)
		self.input.setTrajGait(traj_gait)

	def __get_new_pose(self):
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
		#self.traj.correct_foot_sliding(foot_drifting)
		
		#self.get_root_transform()
		#self.set_previous_pose()
		self.char.set_pose(joint_positions, joint_velocities, joint_rotations)

