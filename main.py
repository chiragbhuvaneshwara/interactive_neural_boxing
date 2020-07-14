# from src.nn.fc_models.mann_tf import MANN as MANNTF
import os

from src.nn.keras_mods.mann_keras import MANN as MANNTF
import json

args_dataset = "./data/boxing_config_updated.json"
# args_output = "./trained_models/mann/"
args_output = "./trained_models/mann/model"

with open(args_dataset) as f:
    config_store = json.load(f)
# datasetnpz = args_dataset.replace(".json", ".npz")

datasetnpz = args_dataset.split('_')[0] + '_train.npz'
# datasetnpz = args_dataset.split('_')[0] + '_test.npz'

# start_poses = (config_store["n_gaits"] + 4) * 12
# n_joints = config_store["numJoints"] * 3
# zero_posture = config_store["zero_posture"]
# middle_point = 6
# window = 12

endJoints = config_store['endJoints']
n_joints = config_store['numJoints']
use_rotations = config_store['use_rotations']
n_gaits = config_store['n_gaits']
use_footcontacts = config_store['use_footcontacts']
foot_left = config_store['foot_left']
foot_right = config_store['foot_right']
zero_posture = config_store['zero_posture']
joint_indices = config_store['joint_indices']
col_inidces = config_store['col_indices']
col_names = config_store['col_names']

x_indices = col_inidces[0]
y_indices = col_inidces[1]

window = 25
middle_point = window // 2


def find_joint_index(name):
    for j in zero_posture:
        if j["name"] == name:
            return j["index"]


# gating_indices = [
#     middle_point + 0 * window, middle_point + 1 * window, middle_point + 2 * window, middle_point + 3 * window, # trajectory
#     4 * window + middle_point + 0 * window, 4 * window + middle_point + 1 * window, 4 * window + middle_point + 2 * window, # gaits
#     # left Foot positions
#     start_poses + find_joint_index("LeftFoot") * 3, start_poses + find_joint_index("LeftFoot") * 3 + 1, start_poses + find_joint_index("LeftFoot") * 3 + 2,
#    #start_poses + find_joint_index("LeftToeBase") * 3, start_poses + find_joint_index("LeftToeBase") * 3 + 1, start_poses + find_joint_index("LeftToeBase") * 3 + 2,
#     # left foot velocities
#     start_poses + n_joints + find_joint_index("LeftFoot") * 3, start_poses + n_joints + find_joint_index("LeftFoot") * 3 + 1, start_poses + n_joints + find_joint_index("LeftFoot") * 3 + 2,
#     #start_poses + n_joints + find_joint_index("LeftToeBase") * 3, start_poses + n_joints + find_joint_index("LeftToeBase") * 3 + 1, start_poses + n_joints + find_joint_index("LeftToeBase") * 3 + 2,
#     # Right Foot positions
#     start_poses + find_joint_index("RightFoot") * 3, start_poses + find_joint_index("RightFoot") * 3 + 1, start_poses + find_joint_index("RightFoot") * 3 + 2,
#     #start_poses + find_joint_index("RightToeBase") * 3, start_poses + find_joint_index("RightToeBase") * 3 + 1, start_poses + find_joint_index("RightToeBase") * 3 + 2,
#     # Right foot velocities
#     start_poses + n_joints + find_joint_index("RightFoot") * 3, start_poses + n_joints + find_joint_index("RightFoot") * 3 + 1, start_poses + n_joints + find_joint_index("RightFoot") * 3 + 2,
#     #start_poses + n_joints + find_joint_index("RightToeBase") * 3, start_poses + n_joints + find_joint_index("RightToeBase") * 3 + 1, start_poses + n_joints + find_joint_index("RightToeBase") * 3 + 2,
# ]

# print(col_names)
# print('\n')
# print(col_inidces)


def generate_indices(indices_dict, key):
    return [i for i in range(indices_dict[key][0], indices_dict[key][1])]


wrist_velocities_indices = generate_indices(x_indices, 'x_right_wrist_vels_tr') + \
                           generate_indices(x_indices, 'x_left_wrist_vels_tr')

current_punch_phase_indices = generate_indices(x_indices, 'x_punch_phase')

punch_target_indices = generate_indices(x_indices, 'x_right_punch_target') + \
                       generate_indices(x_indices, 'x_left_punch_target')

foot_end_effector_velocities = generate_indices(x_indices, 'x_local_vel')[
                               joint_indices['RightFoot']*3:joint_indices['RightFoot']*3 + 3
                               ] + \
                               generate_indices(x_indices, 'x_local_vel')[
                               joint_indices['LeftFoot']*3:joint_indices['LeftFoot']*3 + 3
                               ]

gating_indices = wrist_velocities_indices + \
                 current_punch_phase_indices + \
                 punch_target_indices + \
                 foot_end_effector_velocities

X, Y, norm = MANNTF.prepare_mann_data(datasetnpz, config_store)
mann_config = MANNTF.train_mann(X, Y, norm, gating_indices, args_output)

mann_config_path = r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\mann_config.json'
with open(mann_config_path, 'w') as outfile:
    json.dump(mann_config, outfile)

norm_path = r"C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\data"

with open(os.path.join(norm_path, 'norm.json'), 'w') as outfile:
    json.dump(norm, outfile)

# MANNTF.from_file(datasetnpz, args_output, 30, config_store, gating_indices=gating_indices)
