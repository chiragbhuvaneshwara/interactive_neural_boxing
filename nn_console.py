import argparse

from src.controlers.pfnn_controller import Controller, PFNNOutput, PFNNInput, Trajectory, Character
from src.nn.fc_models.pfnn_np import PFNN
from src.nn.fc_models.pfnn_tf import PFNN as PFNNTF
from src.servers.simpleThriftServer.simpleThriftServer import CREATE_MOTION_SERVER
import numpy as np
import json


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This is the main file to run this project from console. You can either train a new network model or execute an existing one.")
	parser.add_argument("-t", "--train", help="Train a network. Please specify the network type. ", choices=["pfnn"])
	parser.add_argument("-x", "--execute", help="Execute a pretrained network. ", choices=["pfnn_np", "pfnn_tf"])
	parser.add_argument("-d", "--dataset", help="Path to the dataset-description file. The dataset is expected to have the same filename.", required=True)
	parser.add_argument("-o", "--output", help="Path-to-Network during execution, path to folder where to place the trained networks during training. ", required = True)
	parser.add_argument("-e", "--epochs", help="Numbers of epochs for training. ", type=int, default=50)
	args = parser.parse_args()

	print("args: ", args)
	if (args.execute is not None):
		# execute network: 
		target_file = args.output #("trained_models/epoch_3.json")
		if args.execute == "pfnn_np":
			pfnn = PFNN.load(target_file)
		elif args.execute == "pfnn_tf":
			pfnn = PFNNTF.load(target_file)
			pfnn.start_tf()

		dataset_config_file = args.dataset #"data/dataset.json"
		with open(dataset_config_file, "r") as f:
			config_store = json.load(f) 

		# config_store = {"endJoints": 0,
		# 		"numJoints":31,
		# 		"use_rotations": False,
		#		"n_gaits":5}

		c = Controller(pfnn, config_store)

		CREATE_MOTION_SERVER(c)
	elif (args.train is not None):
		with open(args.dataset) as f:
			config_store = json.load(f)
		datasetnpz = args.dataset.replace(".json", ".npz")
		PFNNTF.from_file(datasetnpz, args.output, args.epochs, config_store)





