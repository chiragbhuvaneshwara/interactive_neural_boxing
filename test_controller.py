from src.controlers.pfnn_controller import Controller, PFNNOutput, PFNNInput, Trajectory, Character
from src.nn.fc_models.pfnn_np import PFNN
from src.servers.simpleThriftServer.simpleThriftServer import CREATE_MOTION_SERVER, read_BVH
import numpy as np
import json

from src import utils

target_file = ("trained_models/epoch_0.json")
pfnn = PFNN.load(target_file)

dataset_config_file = "trained_models/dataset.json"
with open(dataset_config_file, "r") as f:
	config_store = json.load(f)

# config_store = {"endJoints": 0,
# 		"numJoints":31,
# 		"use_rotations": False,
#		"n_gaits":5}

c = Controller(pfnn, config_store)

# d = c.pre_render(np.array([0,0,1]), 0.1)
# c.post_render()

#CREATE_MOTION_SERVER(c, r"D:\code\MOSI\utils\test_files\LocomotionFlat01_000.bvh")
#read_BVH(r"D:\code\MOSI\utils\test_files\LocomotionFlat01_000.bvh")

class RawDataController(Controller):

	def __init__(self, network, training_file, config_store):
		#"data/dataset.npz"
		self.frame = 0
		self.network = network
		tmp_d = np.load(training_file)
		self.data = tmp_d["Xun"]
		self.ydata = tmp_d["Yun"]
		self.pdata = tmp_d["Pun"]
		
		self.endJoints = config_store["endJoints"]
		self.n_joints = config_store["numJoints"] + self.endJoints
		self.n_gaits = config_store["n_gaits"]
		self.use_rotations = config_store["use_rotations"]
		self.use_rotations = config_store["use_rotations"]
		self.use_foot_contacts = config_store["use_footcontacts"]
		self.zero_posture = config_store["zero_posture"]


		self.xdim = len(self.data[0])
		self.ydim = len(self.ydata[0])
		print("lenghts: ", self.xdim, self.ydim)
		input_data = np.array([0.0] * self.xdim)
		out_data = np.array([0.0] * self.ydim)
		self.input = PFNNInput(input_data, self.n_joints, self.n_gaits, self.endJoints)
		self.output = PFNNOutput(out_data, self.n_joints, self.endJoints)
		
		self.lastphase = 0
		self.target_vel = np.array((0.0, 0.0, 0.0))

		self.target_dir = np.array((0.0, 0.0, 0.0))
		self.traj = Trajectory()
		self.char = Character(config_store)

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
		
		self.set_previous_pose()


	def pre_render(self, direction, phase):
		
		[out, phase] = self.network.forward_pass([self.data[self.frame], round(self.pdata[self.frame],2)])
		self.output.data = out
		od2 = (self.ydata[self.frame] - self.network.norm["Ymean"]) / self.network.norm["Ystd"]
		out_norm = (out - self.network.norm["Ymean"]) / self.network.norm["Ystd"]
		print("prediction error: ", np.mean(((out_norm - od2)) ** 2))
		#self.output.data = self.ydata[self.frame]
		self.frame += 1
		self.get_root_transform()
		self.get_new_pose()
		return phase
		# set joint positions in self.char

	def post_render(self):
		# do nothing
		return

c = RawDataController(pfnn, "data/dataset.npz", config_store)
CREATE_MOTION_SERVER(c, r"D:\code\MOSI\utils\test_files\LocomotionFlat01_000.bvh")