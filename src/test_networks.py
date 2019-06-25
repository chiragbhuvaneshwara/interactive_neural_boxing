from nn.fc_models.pfnn_tf import PFNN
import numpy as np

if __name__ == "__main__":
	PFNN.from_file("data/dataset.npz", "trained_models/", 30)