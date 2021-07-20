import copy
from flask import Flask, request
import json, os
import math

from vis.backend.controller.boxing.controller import BoxingController
from train.nn.mann_keras.utils import load_mann, load_binary
from vis.backend.server.flask.utils import *

print(os.getcwd())
app = Flask(__name__)

frd = 1
window_wrist = math.ceil(5 / frd)
window_root = math.ceil(5 / frd)
epochs = 100
DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", )
frd_win = 'fr_' + str(frd) + '_tr_' + str(window_root) + "_" + str(window_wrist)
controller_in_out_dir = os.path.join("backend", "controller", "controller_in_out")
frd_win_epochs = frd_win + '_ep_' + str(epochs)
all_models_path = os.path.join("train", "models", "mann_tf2_v2")
# trained_base_path = os.path.join(all_models_path, frd_win_epochs, "2021-05-18_19-35-24", "epochs", "epoch_99")
# trained_base_path = os.path.join(all_models_path, frd_win_epochs, "2021-06-17_19-39-35", "epochs", "epoch_99")
trained_base_path = os.path.join(all_models_path, frd_win_epochs, "2021-07-20_13-40-51", "epochs", "epoch_99")
target_file = os.path.join(trained_base_path, 'saved_model')

x_mean, y_mean = load_binary(os.path.join(trained_base_path, "means", "Xmean.bin")), \
                 load_binary(os.path.join(trained_base_path, "means", "Ymean.bin"))
x_std, y_std = load_binary(os.path.join(trained_base_path, "means", "Xstd.bin")), \
               load_binary(os.path.join(trained_base_path, "means", "Ystd.bin"))

norm = {
    'x_mean': x_mean,
    'y_mean': y_mean,
    'x_std': x_std,
    'y_std': y_std
}

mann = load_mann(os.path.join(trained_base_path, "saved_model"))

dataset_config_path = os.path.join(DATASET_OUTPUT_BASE_PATH, frd_win, "dataset_config.json")
dataset_config_path = os.path.join(dataset_config_path)

with open(dataset_config_path) as f:
    dataset_configuration = json.load(f)

bc = BoxingController(mann, dataset_configuration, norm)
zp = build_zero_posture(bc, num_traj_pts=dataset_configuration["num_traj_samples"])


# print(zp.bones)
# print(zp.bone_map)


def controller_to_posture():
    """
    This function converts the various character and trajectory information stored in the character and trajectory
    classes associated with the controller to an instance of the posture class which can be parsed and visualized on the
    Unity frontend.

    @return: Tposture instance for current frame's posture
    """
    posture = copy.deepcopy(zp)
    pose = bc.get_pose()
    tr = bc.get_trajectroy_for_vis()
    for i in range(len(pose)):
        posture.bones[i].position = np_to_tvector3(pose[i])

    root_pos, root_rot = bc.get_world_pos_rot()

    tr_keys = ['rt', 'rt_v', 'rwt', 'lwt', 'rwt_v', 'lwt_v']
    for i in range(len(tr_keys)):
        curr_tr = tr[i]
        # print(tr_keys[i], len(curr_tr))
        for j in range(len(curr_tr)):
            posture.traj[tr_keys[i]][j] = np_to_tvector3(curr_tr[j])

    posture.location = np_to_tvector3(root_pos)
    posture.rotation = root_rot
    return posture


@app.route('/fetch_frame', methods=['GET', 'POST'])
def fetch_frame():
    """
    This function is associated with a Flask route to obtain the posture information at the current frame.

    @return: str, json data of the posture is sent as string to Unity
    """
    if request.method == 'POST':
        punch_in = request.get_json()

        dir = punch_in["movement_dir"]
        dir[1] *= -1

        punch_hand = punch_in["hand"]
        traj_reached = punch_in["target_reached"]
        # print("---", traj_reached)
        punch_right_target = tvector3_to_np(punch_in["target_right"])
        punch_left_target = tvector3_to_np(punch_in["target_left"])

        if sum(punch_right_target) == 0 and sum(punch_left_target) == 0:
            label = [0, 0]
        elif sum(punch_right_target) != 0 and sum(punch_left_target) == 0:
            label = [1, 0]
        elif sum(punch_right_target) == 0 and sum(punch_left_target) != 0:
            label = [0, 1]

        punch_target = punch_right_target + punch_left_target
        bc.pre_render(punch_target, label, dir, traj_reached, space='global')
        posture = controller_to_posture()
        bc.post_render()
        json_str = json.dumps(posture, default=serialize)

        return json_str

    else:
        print("Problem")


@app.route('/fetch_punch_completed/<target_hand>', methods=['GET'])
def get_punch_completed(target_hand):
    # if target_hand == 'left':
    #     print(target_hand)
    if request.method == 'GET':
        if target_hand == "left":
            p_comp = bc.traj.wrist_reached_left_wrist
        elif target_hand == "right":
            p_comp = bc.traj.wrist_reached_right_wrist
        else:
            p_comp = False
        return json.dumps(p_comp, default=serialize)

    else:
        print("Problem")


@app.route('/fetch_zp', methods=['GET', 'POST'])
def get_zero_posture():
    """
    This function is associated with a flask route to send the initial posture to be animated on Unity. Executed at the
    beginning of the visualization cycle.

    @return: str, json data of the posture is sent as string to Unity
    """
    if request.method == 'POST':
        print(request.get_json()["name"])
        if request.method == 'POST' and request.get_json()["name"] == "fetch_zp":
            posture = controller_to_posture()
            return json.dumps(posture, default=serialize)
        elif request.method == 'POST' and request.get_json()["name"] == "fetch_zp_reset":
            posture = controller_to_posture()
            # bc.reset()
            # bc.reset([0, 0, 0], math.pi, [0, 0, 0])
            bc.reset([0, 0, 0], 0.0, [0, 0, 0])
            # bc.reset([0, 0, 0], 0.0, [0, 0, 1])
            return json.dumps(posture, default=serialize)
    elif request.method == 'GET':
        posture = controller_to_posture()
        # bc.reset()
        # bc.reset([0, 0, 0], -math.pi, [0, 0, 0])
        bc.reset([0, 0, 0], 0.0, [0, 0, 0])
        # bc.reset([0, 0, 0], 0.0, [0, 0, 1])
        return json.dumps(posture, default=serialize)

    else:
        print("Problem")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
