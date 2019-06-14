
from .gen_code.ttypes import TPosture, TBone, TVector3, TGait
from .gen_code import T_simple_directional_motion_server
from ...controlers.pfnn_controller import Controller, Character

from thrift import Thrift
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import threading
import numpy as np
import copy

def read_BVH(file):
	lines = []
	with open(file) as f:
		line = f.readline()
		while ("MOTION" not in line):
			if not line.strip() == "":
				lines.append(line.strip())
			line = f.readline()
		bonelinst = []
		bid = 0
		mapping = {}
		current_bone_name = ""
		current_offset = np.array([0.0,0.0,0.0])
		last_bone_name = ""
		i = 0
		while(i < len(lines)):
			# if "HIERARCHY" in lines[i] or "CHANNELS" in lines[i] or "{" in lines[i]:
			# 	continue
			if "End Site" in lines[i]:
				i+= 4
				continue
			elif "}" in lines[i]:
				last_bone_name = bonelinst[mapping[last_bone_name]].parent

			elif "JOINT" in lines[i] or "ROOT" in lines[i]:
				params = lines[i].split()
				current_bone_name = params[1].strip()

			elif "OFFSET" in lines[i]:
				if current_bone_name != "":
					params = lines[i].split()
					current_offset = TVector3(float(params[1]), float(params[2]), float(params[3]))
					tb = TBone(current_bone_name, current_offset, children=[], parent=last_bone_name)
					bonelinst.append(tb)
					mapping[current_bone_name] = bid
					bid += 1
					last_bone_name = current_bone_name
					current_bone_name = ""
			i += 1
		
		for tb in bonelinst:
			if tb.parent != "":
				pid = mapping[tb.parent]
				bonelinst[pid].children.append(tb.name)
	return TPosture(bonelinst, mapping, TVector3(0,0,0), 0.0)
	
def TVector3_2np(x):
	return np.array([x.X, x.Y, x.Z])
def np_2TVector3(x):
	return TVector3(x[0], x[1], x[2])

class MotionServer:
	def __init__(self, controller: Controller, bvh_path):
		self.log = {}
		self.controller = controller
		self.bvh_path = ""
		self.zero_posture = read_BVH(bvh_path)


	def getZeroPosture(self):
		return self.zero_posture

	def fetchFrame(self, time : float, currentPosture : TPosture, direction : TVector3, gait : TGait):
		newphase = self.controller.lastphase + self.controller.output.getdDPhase()
		print("generating new frame for: ", time, newphase)
		self.controller.pre_render(TVector3_2np(direction), newphase)
		posture = self.__char2TPosture()
		self.controller.post_render()
		return posture
	
	def __char2TPosture(self):
		posture = copy.deepcopy(self.zero_posture)
		char = self.controller.char
		for i in range(len(char.joint_positions)):
			posture.bones[i].Position = np_2TVector3(char.joint_positions[i])
		posture.location = np_2TVector3(char.root_position)
		posture.rotation = char.root_rotation
		return posture


def CREATE_MOTION_SERVER(controller, bvh_path):
	handler = MotionServer(controller, bvh_path)
	processor = T_simple_directional_motion_server.Processor(handler)
	transport = TSocket.TServerSocket(host="127.0.0.1", port=9999)
	tfactory = TTransport.TBufferedTransportFactory()
	pfactory = TBinaryProtocol.TBinaryProtocolFactory()

	server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
	
	thread = threading.Thread(target=server.serve)
	thread.daemon = True
	thread.start()
	thread.join()

	

	# server.serve()
	# server.stop()