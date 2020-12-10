#python nn_console.py -v vinn_tf -d .\data\data_4D_new.json -o .\trained_models\vinn_eval_new\vinn_1\epoch_49.json -ed .\data\data_4D_new_test.npz

import numpy as np
from src.nn.fc_models.vinn_tf import VINN as VINNTF



def run_evaluation(name):
	target_file = r".\trained_models\vinn_eval_new\%s\epoch_49.json"%name
	test_data = r".\data\data_4D_new_test.npz"
	data = np.load(test_data)
	X = data["Xun"]
	Y = data["Yun"]
	P = data["Pun"]

	vinn = VINNTF.load(target_file, batch_size = 25)
	vinn.start_tf()
	acc = vinn.test(X, Y, P, 100, batch_size = 25)

	print(name)
	print("   mean error: ", np.mean(acc))
	print("   mean var:   ", np.mean(np.var(acc, axis=-1)))


run_evaluation("vinn_1")
run_evaluation("vinn_2")
run_evaluation("vinn_3")
run_evaluation("vinn_1n2")
run_evaluation("vinn_1n3")
run_evaluation("vinn_2n3")
run_evaluation("vinn_1n2n3")