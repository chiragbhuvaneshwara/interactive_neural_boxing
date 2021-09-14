import os
import random
import numpy as np
import json

import eval.gen_punch_targets.utils.common as utils

random.seed(17)


def gen_random_punch_targets(x_range, y_range, z_range, save_path, hand):
    punch_targets = []
    for i in range(NUM_POINTS):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)
        punch_targets.append([x, y, z])

    with open(os.path.join(save_path, "random_punch_targets_" + hand + ".json"), 'w') as f:
        json.dump(punch_targets, f)

    return punch_targets


NUM_POINTS = 8
OUTPUT_BASE_PATH = os.path.join("eval", "saved")

OFFSET = 0

saved_punch_targets_json = os.path.join(OUTPUT_BASE_PATH, "targets", "data",
                                        "dataset_punch_targets_local_pos_py_space.json")
random_targets_save_path = os.path.join(OUTPUT_BASE_PATH, "targets", "test")
random_targets_plots_save_path = os.path.join(OUTPUT_BASE_PATH, "plots")

with open(saved_punch_targets_json) as json_file:
    punch_data = json.load(json_file)

punch_targets_dataset_right = np.array(punch_data["punch_targets_dataset_right"])
punch_targets_dataset_left = np.array(punch_data["punch_targets_dataset_left"])

xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_left, OFFSET)
random_punch_targets = gen_random_punch_targets(xR, yR, zR, random_targets_save_path, "left")
utils.plot_punch_targets(random_punch_targets, xR, yR, zR, random_targets_plots_save_path, "left", "random")

xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_right, OFFSET)
random_punch_targets = gen_random_punch_targets(xR, yR, zR, random_targets_save_path, "right")
utils.plot_punch_targets(random_punch_targets, xR, yR, zR, random_targets_plots_save_path, "right", "random")
