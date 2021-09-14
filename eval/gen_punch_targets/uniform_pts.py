import json
import os

from eval.gen_punch_targets.utils.cube_grid import cube_grid_points, cube_grid_count
import numpy as np
import eval.gen_punch_targets.utils.common as utils


def gen_uniform_punch_targets(x_range, y_range, z_range, save_path, hand):
    xv = utils.setup_cube_end_pts(x_range, y_range, z_range)
    punch_targets = cube_grid_points(NS, xv, NG).tolist()

    with open(os.path.join(save_path, "uniform_punch_targets_" + hand + ".json"), 'w') as f:
        json.dump(punch_targets, f)

    return punch_targets


NUM_POINTS = 8
NS = np.zeros(3, dtype=np.int32);
NS[0] = NUM_POINTS ** (1 / 3) - 1
NS[1] = NUM_POINTS ** (1 / 3) - 1
NS[2] = NUM_POINTS ** (1 / 3) - 1

OFFSET = 0

NG = cube_grid_count(NS)
print('')
print('  Number of grid points will be %d' % (NG))

OUTPUT_BASE_PATH = os.path.join("eval", "saved")
saved_punch_targets_json = os.path.join(OUTPUT_BASE_PATH, "targets", "data",
                                        "dataset_punch_targets_local_pos_py_space.json")
uniform_targets_save_path = os.path.join(OUTPUT_BASE_PATH, "targets", "test")
uniform_targets_plots_save_path = os.path.join(OUTPUT_BASE_PATH, "plots")

with open(saved_punch_targets_json) as json_file:
    punch_data = json.load(json_file)

punch_targets_dataset_right = np.array(punch_data["punch_targets_dataset_right"])
punch_targets_dataset_left = np.array(punch_data["punch_targets_dataset_left"])

xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_left, OFFSET)
uniform_punch_targets = gen_uniform_punch_targets(xR, yR, zR, uniform_targets_save_path, "left")
utils.plot_punch_targets(uniform_punch_targets, xR, yR, zR, uniform_targets_plots_save_path, "left", "uniform")

xR, yR, zR = utils.get_mins_maxs(punch_targets_dataset_right, OFFSET)
uniform_punch_targets = gen_uniform_punch_targets(xR, yR, zR, uniform_targets_save_path, "right")
utils.plot_punch_targets(uniform_punch_targets, xR, yR, zR, uniform_targets_plots_save_path, "right", "uniform")
