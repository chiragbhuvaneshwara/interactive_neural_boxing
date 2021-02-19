from src.nn.keras_mods.mann_keras import MANN as MANNTF
# from src.nn.fc_models.mann_tf import MANN as MANNTF
import json
import os
import glob


def generate_indices(indices_dict, key):
    return [i for i in range(indices_dict[key][0], indices_dict[key][1])]


def set_awinda_params():
    right_foot_name = 'RightAnkle'
    left_foot_name = 'LeftAnkle'
    return right_foot_name, left_foot_name


def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    print('Created:', path)


epochs = 100
# epochs = 20
frd = 1
# window = 25
window = 15
#TODO Check norm used
frd_win = 'boxing_fr_' + str(frd) + '_' + str(window)
args_dataset = os.path.join("data", frd_win, "config.json")
with open(args_dataset) as f:
    config_store = json.load(f)

# endJoints = config_store['endJoints']
# n_joints = config_store['numJoints']
# use_rotations = config_store['use_rotations']
# n_gaits = config_store['n_gaits']
# use_footcontacts = config_store['use_footcontacts']
# foot_left = config_store['foot_left']
# foot_right = config_store['foot_right']
# zero_posture = config_store['zero_posture']
# col_names = config_store['col_names']
# frd = config_store['frd']
# window = config_store['window']
joint_indices = config_store['joint_indices']
col_inidces = config_store['col_indices']

x_indices = col_inidces[0]
# y_indices = col_inidces[1]

# frd_win = str(frd) + '_' + str(window)
datasetnpz = os.path.join('data/', frd_win, 'train.npz')
frd_win_epochs = frd_win + '_' + str(epochs)

output_base_path = "trained_models/mann_tf2/"
if frd_win_epochs not in os.listdir(output_base_path):
    print('Creating new output dir')
    os.mkdir(os.path.join(output_base_path, frd_win_epochs))
    out_dir = os.path.join(output_base_path, frd_win_epochs)
else:
    print('Emptying output dir')
    files = glob.glob(os.path.join(output_base_path, frd_win_epochs, '*'))
    for f in files:
        os.remove(f)
    out_dir = os.path.join(output_base_path, frd_win_epochs)

model_wts = os.path.join(out_dir, "model_weights_std_in_out.zip")
mann_config_path = os.path.join(out_dir, 'mann_config.json')
norm_path = os.path.join(out_dir, 'norm.json')

right_foot_key, left_foot_key = set_awinda_params()

wrist_velocities_indices = generate_indices(x_indices, 'x_right_wrist_vels_tr') + \
                           generate_indices(x_indices, 'x_left_wrist_vels_tr')

current_punch_phase_indices = generate_indices(x_indices, 'x_punch_phase')

punch_target_indices = generate_indices(x_indices, 'x_right_punch_target') + \
                       generate_indices(x_indices, 'x_left_punch_target')

foot_end_effector_velocities = generate_indices(x_indices, 'x_local_vel')[
                               joint_indices[right_foot_key] * 3:joint_indices[right_foot_key] * 3 + 3
                               ] + \
                               generate_indices(x_indices, 'x_local_vel')[
                               joint_indices[left_foot_key] * 3:joint_indices[left_foot_key] * 3 + 3
                               ]

gating_indices = wrist_velocities_indices + \
                 current_punch_phase_indices + \
                 punch_target_indices + \
                 foot_end_effector_velocities

X, Y, norm = MANNTF.prepare_mann_data(datasetnpz, config_store)
mann_config = MANNTF.train_mann(X, Y, norm, gating_indices, model_wts, epochs)

save_json(mann_config_path, mann_config)
save_json(norm_path, norm)
