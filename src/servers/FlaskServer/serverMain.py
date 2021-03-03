import copy

from flask import Flask, request
from src.controlers.boxing.controller import BoxingController
import json, os
from src.nn.keras_mods.mann_keras import MANN as MANNTF
from flask import jsonify
from src.servers.FlaskServer.utils import *

app = Flask(__name__)

frd = 1
# window = 25
window = 15
# epochs = 60
epochs = 100
server_to_main_dir = '../../..'
controller_in_out_dir = 'src/controlers/boxing/controller_in_out'
frd_win_epochs = 'boxing_fr_' + str(frd) + '_' + str(window) + '_' + str(epochs)
trained_base_path = 'trained_models/mann_tf2/' + frd_win_epochs
target_file = os.path.join(server_to_main_dir, trained_base_path, 'model_weights_std_in_out.zip')
mann_config_path = os.path.join(server_to_main_dir, trained_base_path, 'mann_config.json')
with open(mann_config_path) as json_file:
    mann_config = json.load(json_file)

mann = MANNTF(mann_config)
mann.load_discrete_weights(target_file)
dataset_config = "data/boxing_fr_" + str(frd) + "_" + str(window) + "/config.json"
dataset_config = os.path.join(server_to_main_dir, dataset_config)

with open(dataset_config) as f:
    config_store = json.load(f)

bc = BoxingController(mann, config_store)
zp = build_zero_posture(bc)

print(zp.bones)
print(zp.bone_map)


def char2TPosture():
    posture = copy.deepcopy(zp)
    pose = bc.get_pose()
    tr = bc.getTrajectroy()
    # arm_tr = bc.getTrajectroy()
    # arm_tr = bc.getGlobalRoot()
    for i in range(len(pose)):
        posture.bones[i].position = np_2TVector3(pose[i])

    root_pos, root_rot = bc.get_world_pos_rot()

    # print('r',arm_tr[0])
    # print('l',arm_tr[1])
    arm_tr_keys = ['rt', 'rt_v', 'rwt', 'lwt', 'rwt_v', 'lwt_v']
    for i in range(len(arm_tr_keys)):
        atr = tr[i]
        for j in range(len(atr)):
            posture.arm_tr[arm_tr_keys[i]][j] = np_2TVector3(atr[j])

    posture.location = np_2TVector3(root_pos)
    posture.rotation = root_rot
    return posture


@app.route('/fetch_frame', methods=['GET', 'POST'])
def fetchFrame():
    if request.method == 'POST':
        # print('Fetching Frame')
        punch_in = request.get_json()
        # print(punch_in)
        punch_hand = punch_in["hand"]
        punch_target = TVector3_2np(punch_in["target_right"]) + TVector3_2np(punch_in["target_left"])
        bc.pre_render(punch_target, space='global')
        posture = char2TPosture()
        bc.post_render()
        json_str = json.dumps(posture, default=serialize)
        print('******************************')
        print(json_str)

        return json_str

    else:
        print("Problem")


@app.route('/fetch_zp', methods=['GET', 'POST'])
def getZeroPosture():
    print(request.get_json()["name"])
    if request.method == 'POST' and request.get_json()["name"] == "fetch_zp":
        posture = char2TPosture()
        return json.dumps(posture, default=serialize)
    elif request.method == 'POST' and request.get_json()["name"] == "fetch_zp_reset":
        posture = char2TPosture()
        bc.reset()
        return json.dumps(posture, default=serialize)
    else:
        print("Problem")


@app.route('/punch_in', methods=['GET', 'POST'])
def post_req():
    if request.method == 'POST':  # this block is only entered when the form is submitted
        # print(request)
        punch_in = request.get_json(force=True)
        # punch_in = request.json
        # print(punch_in)
        punch_hand = punch_in["hand"]
        punch_target = punch_in["target_right"] + punch_in["target_left"]
        # punch_target = punch_target * 100
        bc.pre_render(punch_target, space='global')
        # pose = bc.char.joint_positions
        pose = bc.char.joint_positions
        # pose = pose * 0.01
        # pose = [1, 2, 3]
        bone_map = {}

        if punch_hand != "none":
            print(pose)

        # print(pose)
        # print(punch_in)
        message = {
            'status': 200,
            'message': 'OK',
            'pose': pose.tolist()
        }
        y = json.dumps(message)

        # print(y)

    return jsonify(message)
    # return jsonify(pose.tolist())
    # return jsonify(pose)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
