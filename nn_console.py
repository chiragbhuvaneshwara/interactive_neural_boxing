import argparse

from src.controlers.pfnn_controller import Controller, PFNNOutput, PFNNInput, Trajectory, Character
from src.nn.fc_models.pfnn_np import PFNN
from src.nn.fc_models.pfnn_tf import PFNN as PFNNTF
from src.nn.fc_models.vinn_tf import VINN as VINNTF
from src.servers.simpleThriftServer.simpleThriftServer import CREATE_MOTION_SERVER
from src.servers.MultiThriftServer.MultiThriftServer import CREATE_MOTION_SERVER as CREATE_MULTI_MOTION_SERVER
import numpy as np
import json


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This is the main file to run this project from console. You can either train a new network model or execute an existing one.")
	parser.add_argument("-t", "--train", help="Train a network. Please specify the network type. ", choices=["pfnn", "vinn"])
	parser.add_argument("-x", "--execute", help="Execute a pretrained network. ", choices=["pfnn_np", "pfnn_tf", "vinn_tf"])
	parser.add_argument("-d", "--dataset", help="Path to the dataset-description file. The dataset is expected to have the same filename.", required=True)
	parser.add_argument("-o", "--output", help="Path-to-Network during execution, path to folder where to place the trained networks during training. ", required = True)
	parser.add_argument("-e", "--epochs", help="Numbers of epochs for training. ", type=int, default=50)
	parser.add_argument("-vrl", "--vinn_replace_layers", type=list, default=[])
	parser.add_argument("-vel", "--vinn_each_layer", type=bool, default=False)
	parser.add_argument("-vrd", "--vinn_random_number_dim", type=int, default=-1)
	parser.add_argument("-mth", "--multithreaded", type=bool, default=False)
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
		elif args.execute == "vinn_tf":
			pfnn = VINNTF.load(target_file)
			pfnn.start_tf()

		dataset_config_file = args.dataset #"data/dataset.json"
		with open(dataset_config_file, "r") as f:
			config_store = json.load(f) 

		# config_store = {"endJoints": 0,
		# 		"numJoints":31,
		# 		"use_rotations": False,
		#		"n_gaits":5}

		c = Controller(pfnn, config_store)

		if args.multithreaded:
			CREATE_MULTI_MOTION_SERVER(c)
		else:
			CREATE_MOTION_SERVER(c)
	elif (args.train is not None):
		if args.train == "pfnn":
			with open(args.dataset) as f:
				config_store = json.load(f)
			datasetnpz = args.dataset.replace(".json", ".npz")
			PFNNTF.from_file(datasetnpz, args.output, args.epochs, config_store)
		elif args.train == "vinn":
			with open(args.dataset) as f:
				config_store = json.load(f)
			datasetnpz = args.dataset.replace(".json", ".npz")
			replace_layers = []
			for l in args.vinn_replace_layers:
				if l == ' ':
					continue
				else:
					replace_layers.append(int(l))

			print(replace_layers)

			random_number_dim = args.vinn_random_number_dim
			each_layer = args.vinn_each_layer

			VINNTF.from_file(datasetnpz, args.output, args.epochs, config_store, replace_layers = replace_layers, draw_each_layer = each_layer, random_number_dim = random_number_dim)






