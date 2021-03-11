from src.nn.keras_mods.mann_keras import MANN as MANNTF
import json
import os
import glob


def generate_id_sequence(column_demarcation_ids, key):
    return [i for i in range(column_demarcation_ids[key][0], column_demarcation_ids[key][1])]


def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    print('Created:', path)


def setup_output_dir(output_base_path, output_directory):
    """
    Sets up output dir if it doesn't exist. If it does exist, it empties the dir.
    :param output_base_path: str
    :param output_directory: str
    """
    if output_directory not in os.listdir(output_base_path):
        print('Creating new output dir:', output_directory)
        os.mkdir(os.path.join(output_base_path, output_directory))
    else:
        print('Emptying output dir:', output_directory)
        files = glob.glob(os.path.join(output_base_path, output_directory, '*'))
        for fi in files:
            os.remove(fi)


def get_gating_indices(x_ids, joint_ids):
    wrist_velocities_tr_ids = generate_id_sequence(x_ids, 'x_right_wrist_vels_tr') + \
                              generate_id_sequence(x_ids, 'x_left_wrist_vels_tr')

    punch_labels_tr_ids = generate_id_sequence(x_ids, 'x_right_punch_labels_tr') + \
                          generate_id_sequence(x_ids, 'x_left_punch_labels_tr')

    current_punch_labels_ids = generate_id_sequence(x_ids, 'x_right_punch_labels') + \
                               generate_id_sequence(x_ids, 'x_left_punch_labels')

    punch_target_ids = generate_id_sequence(x_ids, 'x_right_punch_target') + \
                       generate_id_sequence(x_ids, 'x_left_punch_target')

    f_r_id = joint_ids["RightAnkle"] * 3
    f_l_id = joint_ids["LeftAnkle"] * 3
    foot_end_effector_velocities_ids = generate_id_sequence(x_ids, 'x_local_vel')[f_r_id: f_r_id + 3] + \
                                       generate_id_sequence(x_ids, 'x_local_vel')[f_l_id: f_l_id + 3]
    w_r_id = joint_ids["RightWrist"] * 3
    w_l_id = joint_ids["LeftWrist"] * 3
    wrist_end_effector_velocities_ids = generate_id_sequence(x_ids, 'x_local_vel')[w_r_id: w_r_id + 3] + \
                                        generate_id_sequence(x_ids, 'x_local_vel')[w_l_id: w_l_id + 3]

    # Gating input for OG MANN:
    # foot end effector velocities,
    # the current action variables
    # and the desired velocity of the character
    gating_ids = wrist_velocities_tr_ids + \
                 punch_labels_tr_ids + \
                 punch_target_ids + \
                 wrist_end_effector_velocities_ids \
                 + \
                 current_punch_labels_ids + \
                 foot_end_effector_velocities_ids
    # desired_vel (part of OG MANN gating input)
    return gating_ids


DEVELOP = False
EPOCHS = 100
FRD = 1
WINDOW = 15
OUT_BASE_PATH = os.path.join("models", "mann_tf2")
############################################
frd_win = 'boxing_fr_' + str(FRD) + '_' + str(WINDOW)

if not DEVELOP:
    # TODO Rename dataset config file in file system to dataset_config.json instead of config.json
    dataset_config_path = os.path.join("data", frd_win, "config.json")
    dataset_npz_path = os.path.join('data', frd_win, 'train.npz')
elif DEVELOP:
    EPOCHS = 2
    OUT_BASE_PATH = os.path.join("models", "mann_tf2", "dev")
    dataset_config_path = os.path.join("data", "dev", frd_win, "config.json")
    dataset_npz_path = os.path.join('data', 'dev', frd_win, 'train.npz')

frd_win_epochs = frd_win + '_' + str(EPOCHS)
setup_output_dir(OUT_BASE_PATH, frd_win_epochs)
# if frd_win_epochs not in os.listdir(output_base_path):
#     print('Creating new output dir')
#     os.mkdir(os.path.join(output_base_path, frd_win_epochs))
#     out_dir = os.path.join(output_base_path, frd_win_epochs)
# else:
#     print('Emptying output dir')
#     files = glob.glob(os.path.join(output_base_path, frd_win_epochs, '*'))
#     for f in files:
#         os.remove(f)
#     out_dir = os.path.join(output_base_path, frd_win_epochs)
out_dir = os.path.join(OUT_BASE_PATH, frd_win_epochs)
model_wts_path = os.path.join(out_dir, "model_weights.zip")
mann_config_path = os.path.join(out_dir, 'mann_config.json')
norm_path = os.path.join(out_dir, 'data_norm.json')

############################################
with open(dataset_config_path) as f:
    dataset_config = json.load(f)
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
bone_map = dataset_config['bone_map']
col_demarcation_ids = dataset_config['col_demarcation_ids']
x_col_demarcation = col_demarcation_ids[0]

gating_indices = get_gating_indices(x_col_demarcation, bone_map)

X, Y, norm = MANNTF.prepare_mann_data(dataset_npz_path, dataset_config)
mann_config = MANNTF.train_mann(X, Y, norm, gating_indices, model_wts_path, EPOCHS)
############################################

save_json(mann_config_path, mann_config)
save_json(norm_path, norm)
