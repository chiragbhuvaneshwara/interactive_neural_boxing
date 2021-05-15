from . import utils
from .bvh import BVHReader
from .skeleton_builder import SkeletonBuilder
from .motion_vector import MotionVector

# import .utils

import scipy.ndimage.filters as filters

import numpy as np
import math

holden_hips = ["LeftUpLeg", "RightUpLeg"]
holden_shoulders = ["LeftArm", "RightArm"]

holden_directions = np.array([
        [0.000000, 0.100000, 0.000000],  # Hips
        [1.363060, -1.794630, 0.839290],  # LHipJoint
        [2.448110, -6.726130, -0.000000],  # LeftUpLeg
        [2.562200, -7.039591, 0.000000],  # LeftLeg
        [0.157640, -0.433110, 2.322551],  # LeftFoot
        [0.000000, 0.100000, 0.000000],  # LeftToeBase
        [-1.305520, -1.794630, 0.839290],  # RHipJoint
        [-2.542531, -6.985552, 0.000001],  # RightUpLeg
        [-2.568260, -7.056230, -0.000000],  # RightLeg
        [-0.164730, -0.452590, 2.363150],  # RightFoot
        [0.000000, 0.100000, 0.000000],  # RightToeBase
        [0.028270, 2.035590, -0.193380],  # LowerBack
        [0.056720, 2.048850, -0.042750],  # Spine
        [0.056720, 2.048850, -0.042750],  # Spine1
        [-0.054170, 1.746240, 0.172020],  # Neck
        [0.104070, 1.761360, -0.123970],  # Neck1
        [0.000000, 0.100000, 0.000000],  # Head
        [3.362410, 1.200890, -0.311210],  # LeftShoulder
        [4.983000, 0.000000, -0.000000],  # LeftArm
        [3.483560, 0.000000, 0.000000],  # LeftForeArm
        [3.483561, 0.000000, 0.000000],  # LeftHand
        [0.715260, 0.000000, 0.000000],  # LeftFingerBase
        [0.000000, 0.100000, 0.000000],  # LeftHandIndex1
        [0.000000, 0.100000, 0.000000],  # LThumb
        [-3.136600, 1.374050, -0.404650],  # RightShoulder
        [-5.241900, 0.000000, -0.000000],  # RightArm
        [-3.444170, 0.000000, 0.000000],  # RightForeArm
        [-3.444170, 0.000000, 0.000000],  # RightHand
        [-0.622530, 0.000000, 0.000000],  # RightFingerBase
        [0.000000, 0.100000, 0.000000],  # RightHandIndex1
        [0.000000, 0.100000, 0.000000]  # RThumb
    ])

class Animation():

	def __init__(self, filename, includes, directions = holden_directions, holden_corrected = False):
		""" Animation is a Wrapper function for the skeleton definitions in morphable-graphs.

		The functions provided with this class enable the data loading and preprocessing of
		BVH motion capture data. Just enter the path to the bvh file and the wrapper will
		deal with the rest

		If the transform is corrected for Holden Data, [0, 8.742816925048828, 0.13204166293144226] is substracted.
		TO DO: Find fix for this hack.

		Args:
			filename (str): Path to bvh file to be read.

		Attributes:
			filename (str): Path to bvh file
			animated_joints (list): List of animated joints

		Authors:
			Janis

		:param filename: Path to bvh file to be read.
		:param holden_corrected: corrects global position to conform to Holden's BVH reader. This is a Hack!

		"""
		self.filename = filename
		self._bvhreader = BVHReader(filename)
		self._motion_vector = MotionVector()
		self._motion_vector.from_bvh_reader(self._bvhreader, False)
		self.animated_joints = list(self._bvhreader.get_animated_joints())
		self._motion_vector.skeleton = SkeletonBuilder().load_from_bvh(self._bvhreader, animated_joints=self.animated_joints)

		self._global_transformations = []
		self._holden_corrected = holden_corrected

		self._root_forward = []
		self._root_rotations = []

		self.directions = holden_directions
		self.twists = []

		self.includes = includes

	def get_global_transform(self, frame):
		"""
		Generates global transformation matrices for a single frame.

		:param frame
		:return: np.array [joints, 4, 4]
		"""
		if len(self._global_transformations) > frame:
			return self._global_transformations[frame]
		globaltransf = []
		for bone_name in self._motion_vector.skeleton.get_joint_names():
			globaltransf.append(self._motion_vector.skeleton.nodes[bone_name].get_global_matrix(self._motion_vector.frames[frame]))
		return np.array(globaltransf)

	def get_global_transforms(self, steplength=1):
		"""
		Generates global transformation matrices for all frames.

		:param steplength:int set to > 1 to scale down frames (steplength = 2 reduces the frames by 2)
		:return: np.array [frames / steplength, joints, 4, 4]
		"""

		if len(self._global_transformations) == 0:
			# recompute transforms
			globaltransf = []
			for f in range(0, self._motion_vector.n_frames, steplength):
				globaltransf.append(self.get_global_transform(f))
			self._global_transformations = np.array(globaltransf)

			if self._holden_corrected:
				# for some reason, y, z components are not correctly set. Correct this, if the bug was found!!
				posUpdate = [0, 8.742816925048828, 0.13204166293144226]
				for f in range(len(self._global_transformations)):
					for b in range(len(self._global_transformations[f])):
						self._global_transformations[f][b][:3, 3] -= posUpdate
			if len(self.includes > 0):
				self._global_transformations = self._global_transformations[self.includes]
		return self._global_transformations

	def get_global_joint_positions(self):
		"""
		Generates vector of global joint positions for whole animation

		:return: np.array [frames, joints, 3]
		"""
		return self.get_global_transforms()[:,:,:3,3] / self.get_global_transforms()[:,:,3:,3]

	def get_global_joint_rotations(self):
		"""
		Generates Vector of global joint rotations for whole animation.
		:return: np.array(frames, joints, 3, 3
		"""
		globals = self.get_global_transforms()
		return globals[:, :, :3, :3]


	def get_forward_directions_rotations(self, hip_joints = holden_hips, shoulder_joints = holden_shoulders, filter_width = 20):
		"""
		Generates and smoothes forward directions for animations by taking the cross vector to the average vector of hip and shoulder joints.

		Root rotations are generated out of the forward directions by computing the transformation matrices between forward direction and [0,0,1].


		:return: forward-directions: np.array [frames, 3]
		:return: root_rotations: np.array [frames, 4,4]


		:param hip_joints: list [left, right] of hip joint names. If no hip joints are available, use upper leg joints
		:param shoulder_joints: list [left, right] of shoulder joint names. If no shoulder joints are available, take upper arm.
		:param filter_width: width of filter for forward vector.
		"""

		if len(self._root_forward) > 0:
			return self._root_forward, self._root_rotations

		# gather indices and global joint positions
		sdr_l, sdr_r = self.animated_joints.index(shoulder_joints[0]),self.animated_joints.index(shoulder_joints[1])
		hip_l, hip_r = self.animated_joints.index(hip_joints[0]), self.animated_joints.index(hip_joints[1])
		global_positions = self.get_global_joint_positions()

		# this was taken from Holden et al., 2017
		# find across vectors
		across = (
				(global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
				(global_positions[:, hip_l] - global_positions[:, hip_r]))
		across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

		# find forward vectors
		forward = filters.gaussian_filter1d(
			np.cross(across, np.array([[0.0, 1.0, 0.0]])), filter_width, axis=0, mode='nearest')
		forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

		root_rotation = utils.matrix_rotation_from_vector_to_vector_frames(forward, np.array([0,0,1]))
		root_rotation = root_rotation[:,:3, :3]
		self._root_forward, self._root_rotations = forward, root_rotation
		return forward, root_rotation

	def get_root_local_joint_position(self):
		"""
		Computes and returns the joint position in root local space for all frames.

		root local space is the space local to the current (framewise) root location and rotation w.r.t. forward direction.

		:return: np.array [frames-1, joints, 3]
		"""
		local_positions = self.get_global_joint_positions().copy()

		# adjust to root location
		local_positions[:, :, 0] -= local_positions[:, 0:1, 0]
		local_positions[:, :, 2] -= local_positions[:, 0:1, 2]

		# adjust to root rotation
		f,r = self.get_forward_directions_rotations()
		r_ = np.reshape(r, (r.shape[0], 1, 3, 3))
		lp_ = np.reshape(local_positions, (local_positions.shape[0], local_positions.shape[1], 3, 1))
		lp_ = np.matmul(r_, lp_)[:-1]

		return lp_

	def get_root_local_joint_velocity(self):
		"""
		Computes and returns the joint velocity in root local space for all frames.

		root local space is the space local to the current (framewise) root location and rotation w.r.t. forward direction.

		:return: np.array [frames-1, joints, 3]
		"""
		# compute velocity
		globals = self.get_global_joint_positions()
		vel = globals[1:] - globals[:-1]

		# adjust to rotation:
		f, r = self.get_forward_directions_rotations()
		r_ = np.reshape(r, (r.shape[0], 1, 3, 3))
		vel = np.reshape(vel, (vel.shape[0], vel.shape[1], 3, 1))
		vel = np.matmul(r_[:-1], vel)
		return vel


	def get_root_local_joint_rotations(self):
		"""
		returns the root local joint rotations matrices
		:return: np.array(frames, bones, 3, 3)
		"""
		global_rot = self.get_global_joint_rotations()
		f, r = self.get_forward_directions_rotations()
		r_ = np.reshape(r, (r.shape[0], 1, 3, 3))
		local_rot = np.matmul(r_, global_rot)
		return local_rot[:-1]

	def __swing_twist_decomposition(self, q, twist_axis):
		"""
		internal method to compute the swing twist decomposition for quaternion q and returns the respective quaternions.
		:param q:
		:param twist_axis:
		:return: swing: np.array(4), twist: np.array(4)
		"""
		twist_axis /= np.sqrt(np.sum(twist_axis**2))

		#twist_axis = np.array((q * offset))[0]
		projection = np.dot(twist_axis, np.array([q[1], q[2], q[3]])) * twist_axis
		twist = (np.array([q[0], projection[0], projection[1],projection[2]], dtype=float))
		if (np.sum(twist**2) == 0):
			twist = (np.array([[1.0,0.0,0.0,0.0]]))

		twist /= np.linalg.norm(twist)
		e = 0.00001
		epsilon = np.array([e,e,e])
		(angle, axis) = utils.quaternion_angle_axis_frames(np.array([twist]))

		if np.all(np.abs(axis + twist_axis) <= epsilon):
			twist = np.array([-twist[0], -twist[1], - twist[2], - twist[3]]).transpose()
		inverted_twist = np.array([-twist[0], -twist[1], - twist[2], - twist[3]]).transpose()
		swing = utils.quaternion_multiply(q, inverted_twist) #q * inverted_twist

		return (swing, twist)

	def get_joint_swing_twist_rotation(self):
		"""
		Transforms joint rotations to swing and twist rotations in root local space

		:return: twists: np.array(frames, bones, 1)
		"""
		if len(self.twists) > 0:
			return np.array(self.twists)

		# normalize forwards
		forwards = np.array(self.directions)
		forwards /= np.reshape(np.linalg.norm(forwards, axis=-1), (-1,1))

		qs = self.get_root_local_joint_rotations()
		twist_axes = np.reshape(np.matmul(qs, np.reshape(forwards, (forwards.shape[0], 3, 1))), (qs.shape[0], qs.shape[1], 3))
		qs = np.array(utils.quaternions_from_matrix(qs))

		frames = []
		for f in range(len(qs)):
			bone = []
			for b in range(qs.shape[1]):

				twist_axis = twist_axes[f][b]#np.matmul(self.get_root_local_joint_rotations()[f][b], forwards[b])
				q = qs[f][b]
				(swing, twist) = self.__swing_twist_decomposition(q, twist_axis)
				(angle, axis) = utils.quaternion_angle_axis_frames(np.array([twist]))
				bone.append(angle[0])
			frames.append(bone)
		self.twists = np.array(frames)
		self.twists[self.twists > math.pi] -= 2 * math.pi
		return np.array(self.twists)



	def get_root_pos(self):
		"""
		Retrieve global root-positions for all frames.

		:return: np.array [frames, 3]
		"""
		global_xforms = self.get_global_transforms()
		return global_xforms[:,0,:3,3] / global_xforms[:,0,3:,3]

	def get_frames(self):
		"""
		Number of frames in animation.
		:return: frames (int)
		"""
		return self._motion_vector.n_frames

	def get_root_rotational_velocity(self):
		"""
		Generates list of rotational velocity angles (rotational change along up axis between 2 frames)
		:return: np.array(frames, 1)
		"""
		f, r = self.get_forward_directions_rotations()
		rots = np.matmul(r[1:], np.linalg.inv(r)[:-1])
		forwards = np.array([0.0, 0.0, 1.0])

		directions = np.matmul(rots, forwards)

		# directions[:, 1] = 0.0
		# directions /= np.reshape(np.linalg.norm(directions, axis=1), (-1, 1))
		angles = np.arctan2(directions[:,0], directions[:,2])
		return np.reshape(angles, (-1, 1))

	def get_root_velocity(self):
		"""
		Generates root velocity for frames (difference of positions between frames) in root_local space)
		:return: np.array(frames, 1, 3)
		"""
		f, r = self.get_forward_directions_rotations()
		vel = np.matmul(r[:-1], np.reshape(self.get_global_joint_positions()[1:, 0] - self.get_global_joint_positions()[:-1, 0], (r.shape[0]-1, 3, 1)))
		return np.reshape(vel, (-1, 1, 3))

	def get_trajectory(self, frame, window = 60):
		"""
		Generate trajectory information (pos, dir) local to current frame
		:param frame:int current frame
		:param window:int window around current frame
		:return: pos: np.array(window // 10 * 2, 3), dir: np.array(window // 10 * 2, 3)
		"""
		f, r = self.get_forward_directions_rotations()
		root_positions = np.matmul(r[frame], np.reshape(self.get_root_pos()[frame - window:frame + window:10] - self.get_root_pos()[frame], (-1, 3, 1)))
		root_dirs = np.matmul(r[frame], np.reshape(f[frame-window:frame+window:10], (-1, 3, 1)))
		return (np.reshape(root_positions, (-1, 3)), np.reshape(root_dirs, (-1,3)))

	def get_root_local_end_joint_pos(self, bone_name, forwards):
		bone_id = self._motion_vector.skeleton.get_joint_names().index(bone_name)

		g = self.get_global_joint_rotations().copy()
		bone_transforms = g[:, bone_id]
		end_joint_dir = np.matmul(bone_transforms, forwards)

		gl = self.get_global_joint_positions()
		end_joint_pos = gl[:, bone_id] + end_joint_dir
		vel = end_joint_pos[1:] - end_joint_pos[:-1]

		end_joint_pos[:, 0] -= gl[:, 0, 0]
		end_joint_pos[:, 2] -= gl[:, 0, 2]



		f, r = self.get_forward_directions_rotations()
		vel = np.matmul(r[:-1], np.reshape(vel, (-1, 3,1)))
		end_joint_pos = np.matmul(r, np.reshape(end_joint_pos, (-1, 3, 1)))[:-1]

		return (bone_id, end_joint_pos, vel)