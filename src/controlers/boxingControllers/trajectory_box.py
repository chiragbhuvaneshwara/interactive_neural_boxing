"""author: Janis Sprenger """
import numpy as np
import math, time
from ...nn.fc_models.fc_networks import FCNetwork
from ... import utils

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
	def __init__(self, config_store):
		self.config_store = config_store

		self.n_traj_samples = self.config_store['num_traj_samples']
		self.traj_step_size = self.config_store['traj_step']

		# 12 * 10 = 120 fps trajectory window or 10 * 5 = 50fps trajectory window
		self.n_frames_tr_win = self.n_traj_samples * self.traj_step_size
		self.median_idx = self.n_frames_tr_win // 2

		n_3d_dims = 3
		unit_vecs = np.eye(n_3d_dims, n_3d_dims)
		z_axis = 2
		z_vec = unit_vecs[z_axis:z_axis+1, :]
		self.traj_positions = np.zeros((self.n_frames_tr_win, n_3d_dims))
		self.traj_directions = np.tile(z_vec, (self.n_frames_tr_win, 1))

		self.foot_drifting = np.zeros(n_3d_dims)
		self.blend_bias = 2.0

		# self.traj_positions = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
		# self.traj_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames_tr_win)
		# self.traj_rotations = np.array([0.0] * self.n_frames_tr_win)

		# self.traj_gait_stand = np.array([1.0] * self.n_frames_tr_win)
		# self.traj_gait_walk = np.array([0.0] * self.n_frames_tr_win)
		# self.traj_gait_jog = np.array([0.0] * self.n_frames_tr_win)
		# self.traj_gait_back = np.array([0.0] * self.n_frames_tr_win)
		# self.foot_drifting = np.array([0.0, 0.0, 0.0])
		# self.blend_bias = 2.0

	def reset(self, start_location = [0,0,0], start_orientation = [1,0,0,0], start_direction = [0,0,1]):
		"""
		Resets the trajectory information and thus the character to 0,0,0 pointing to 0,0,1. 
		
		Keyword Arguments:
			start_location {list} -- [description] (default: {[0,0,0]})
			start_orientation {list} -- [description] (default: {[1,0,0,0]})
			start_direction {list} -- [description] (default: {[0,0,1]})
		"""
		self.traj_positions = np.array([[0.0, 0.0, 0.0]] * self.n_frames_tr_win)
		self.traj_directions = np.array([[0.0, 0.0, 1.0]] * self.n_frames_tr_win)
		self.traj_rotations = np.array([0.0] * self.n_frames_tr_win)
		for idx in range(self.median_idx+1):
			self.traj_positions[idx] = np.array(start_location)
			self.traj_rotations[self.median_idx] = np.array(start_orientation)
			self.traj_directions[self.median_idx] = np.array(start_direction)

		self.traj_gait_stand = np.array([0.0] * self.n_frames_tr_win)
		for idx in range(self.median_idx + 1):
			self.traj_gait_stand[idx] = 1
		self.traj_gait_walk = np.array([0.0] * self.n_frames_tr_win)
		self.traj_gait_jog = np.array([0.0] * self.n_frames_tr_win)
		self.traj_gait_back = np.array([0.0] * self.n_frames_tr_win)

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
		w = self.n_frames_tr_win // 10  # only every 10th frame => resulting in w = 12
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
		#for f in range(self.median_idx, self.n_frames_tr_win, 10):
		#    frames_tmp_future.append(self.traj_positions[f])

		for i in range(0, self.n_frames_tr_win, 10):
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
		for i in range(0, self.n_frames_tr_win, 10):
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
		for i in range(0, self.n_frames_tr_win):
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
		for i in range(self.median_idx + 1, self.n_frames_tr_win):
			w = self.median_idx // 10
			m = ((i - self.n_frames_tr_win / 2) / 10.0) % 1.0
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

	def getWorldPosRot(self):
		pos = np.array(self.traj_positions[self.median_idx])
		pos[1] = 0.0

		rot = self.traj_rotations[self.median_idx]
		return (pos, rot)

	def getPreviousPosRot(self):
		pos = np.array(self.traj_positions[self.median_idx -1])
		pos[1] = 0.0

		rot = self.traj_rotations[self.median_idx -1]
		return (pos, rot)
