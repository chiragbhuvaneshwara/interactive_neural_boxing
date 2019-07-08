from nn.fc_models.pfnn_tf import PFNN
import numpy as np
import json

if __name__ == "__main__":
	with open("data/dataset.json") as f:
		config_store = json.load(f)
	PFNN.from_file("data/dataset.npz", "trained_models/", 50, config_store)