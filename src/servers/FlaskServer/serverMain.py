import copy

from flask import Flask, request
from src.controlers.boxing.controller import BoxingController
import json, os
from src.nn.keras_mods.mann_keras import MANN as MANNTF
from flask import jsonify
from src.servers.FlaskServer.utils import *

# TODO setup better names for variables used in Flask server
print(os.getcwd())
app = Flask(__name__)

frd = 1
window = 15
epochs = 100
# server_to_main_dir = '../../../'
server_to_main_dir = ''
DATASET_OUTPUT_BASE_PATH = server_to_main_dir + 'data/'
frd_win = 'boxing_fr_' + str(frd) + '_' + str(window)
dataset_path = os.path.join(DATASET_OUTPUT_BASE_PATH, frd_win, 'train.npz')
controller_in_out_dir = 'src/controlers/boxing/controller_in_out'
frd_win_epochs = 'boxing_fr_' + str(frd) + '_' + str(window) + '_' + str(epochs)
trained_base_path = 'saved_models/mann_tf2/' + frd_win_epochs
target_file = os.path.join(server_to_main_dir, trained_base_path, 'model_weights.zip')
mann_config_path = os.path.join(server_to_main_dir, trained_base_path, 'mann_config.json')
with open(mann_config_path) as json_file:
    mann_config = json.load(json_file)

mann = MANNTF(mann_config)
mann.load_discrete_weights(target_file)
dataset_config = "data/boxing_fr_" + str(frd) + "_" + str(window) + "/config.json"
dataset_config = os.path.join(server_to_main_dir, dataset_config)

with open(dataset_config) as f:
    config_store = json.load(f)

bc = BoxingController(mann, config_store, dataset_path)
zp = build_zero_posture(bc)

print(zp.bones)
print(zp.bone_map)


def controller_to_posture():
    posture = copy.deepcopy(zp)
    pose = bc.get_pose()
    tr = bc.get_trajectroy_for_vis()
    for i in range(len(pose)):
        posture.bones[i].position = np_to_tvector3(pose[i])

    root_pos, root_rot = bc.get_world_pos_rot()

    tr_keys = ['rt', 'rt_v', 'rwt', 'lwt', 'rwt_v', 'lwt_v']
    for i in range(len(tr_keys)):
        curr_tr = tr[i]
        for j in range(len(curr_tr)):
            posture.traj[tr_keys[i]][j] = np_to_tvector3(curr_tr[j])

    posture.location = np_to_tvector3(root_pos)
    posture.rotation = root_rot
    return posture


@app.route('/fetch_frame', methods=['GET', 'POST'])
def fetch_frame():
    # TODO First time of fetching wrist pos tr is incorrect ==>

    if request.method == 'POST':
        punch_in = request.get_json()
        punch_hand = punch_in["hand"]

        punch_right_target = tvector3_to_np(punch_in["target_right"])
        punch_left_target = tvector3_to_np(punch_in["target_left"])

        print(punch_left_target)
        print(punch_right_target)

        # TODO POST punch_labels from the backend
        if sum(punch_right_target) == 0 and sum(punch_left_target) == 0:
            label = [0, 0]
        elif sum(punch_right_target) != 0 and sum(punch_left_target) == 0:
            label = [1, 0]
        elif sum(punch_right_target) == 0 and sum(punch_left_target) != 0:
            label = [0, 1]

        punch_target = punch_right_target + punch_left_target
        bc.pre_render(punch_target, label, space='global')
        posture = controller_to_posture()
        bc.post_render()
        json_str = json.dumps(posture, default=serialize)

        return json_str

    else:
        print("Problem")


@app.route('/fetch_zp', methods=['GET', 'POST'])
def get_zero_posture():
    print(request.get_json()["name"])
    if request.method == 'POST' and request.get_json()["name"] == "fetch_zp":
        posture = controller_to_posture()
        return json.dumps(posture, default=serialize)
    elif request.method == 'POST' and request.get_json()["name"] == "fetch_zp_reset":
        posture = controller_to_posture()
        # TODO Check reset
        bc.reset()
        return json.dumps(posture, default=serialize)
    else:
        print("Problem")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
