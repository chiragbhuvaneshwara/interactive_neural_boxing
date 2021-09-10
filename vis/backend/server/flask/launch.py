import copy
import json
import math
import os

from flask import Flask, request, jsonify

from train.nn.mann_keras.utils import load_mann, load_binary
from vis.backend.controller.boxing.controller import BoxingController
from vis.backend.server.flask.utils import *

print(os.getcwd())
app = Flask(__name__)

frd = 1
window_wrist = math.ceil(5 / frd)
window_root = math.ceil(6 / frd)
epochs = 100

# DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", )
DATASET_OUTPUT_BASE_PATH = os.path.join("data", "neural_data", "dev")

frd_win = 'fr_' + str(frd) + '_tr_' + str(window_root) + "_" + str(window_wrist)
controller_in_out_dir = os.path.join("backend", "controller", "controller_in_out")
frd_win_epochs = frd_win + '_ep_' + str(epochs)
all_models_path = os.path.join("train", "models", "mann_tf2_v2")
###################################################################################################
# trained_base_path = os.path.join(all_models_path, frd_win_epochs, "2021-08-04_17-39-31", "epochs",
#                                  "epoch_99")  # 1, 5, 5 min gating inputs + traj root dirs ==> full traj
###################################################################################################

trained_base_path = os.path.join("train/models/mann_tf2_v2/dev/fr_1_tr_6_5_ep_2/2021-09-09_14-10-39", "epochs",
                                 "epoch_1")
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

# eval_save_path = os.path.join(os.sep.join(trained_base_path.split(os.sep)[:-2]), "eval_{n_punches}.csv")
eval_save_path = os.path.join(os.sep.join(trained_base_path.split(os.sep)[:-2]), "{eval_csv_name}")

with open(dataset_config_path) as f:
    dataset_configuration = json.load(f)

bc = BoxingController(mann, dataset_configuration, norm)
zp = build_zero_posture(bc, num_traj_pts_root=dataset_configuration["num_traj_samples_root"],
                        num_traj_pts_wrist=dataset_configuration["num_traj_samples_wrist"]
                        )

n_punches_eval = 0
eval_csv_name = ""


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
        facing_dir = punch_in["facing_dir"]

        punch_hand = punch_in["hand"]
        traj_reached = punch_in["target_reached"]
        punch_right_target = tvector3_to_np(punch_in["target_right"])
        punch_left_target = tvector3_to_np(punch_in["target_left"])

        if sum(punch_right_target) == 0 and sum(punch_left_target) == 0:
            label = [0, 0]
        elif sum(punch_right_target) != 0 and sum(punch_left_target) == 0:
            label = [1, 0]
        elif sum(punch_right_target) == 0 and sum(punch_left_target) != 0:
            label = [0, 1]

        punch_target = punch_right_target + punch_left_target
        bc.pre_render(punch_target, label, dir, facing_dir, traj_reached, space='global')
        posture = controller_to_posture()
        bc.post_render()
        json_str = json.dumps(posture, default=serialize)

        return json_str

    else:
        print("Problem")


@app.route('/set_n_punches_eval/<n_punches>', methods=['POST'])
def set_n_punches_eval(n_punches):
    """

    @return:
    """
    if request.method == 'POST':
        global n_punches_eval
        n_punches_eval = n_punches

        return jsonify(success=True)

    else:
        print("Problem")


@app.route('/set_eval_name/',
           methods=['POST'])
def set_eval_name():
    """

    @return:
    """
    if request.method == 'POST':
        global eval_csv_name
        # exp_duration_indicator = str(exp_duration_indicator)
        eval_type = request.args.get("eval_type")
        exp_type = request.args.get("exp_type")
        exp_duration_indicator = request.args.get("exp_duration_indicator")
        eval_csv_name = "eval_" + "_".join([eval_type, exp_type, exp_duration_indicator]) + ".csv"

        return jsonify(success=True)

    else:
        print("Problem")


@app.route('/fetch_punch_completed/<target_hand>', methods=['GET'])
def get_punch_completed(target_hand):
    if request.method == 'GET':
        if target_hand == "left":
            p_comp = bc.traj.punch_completed_left
            p_h_comp = bc.traj.punch_half_completed_left
        elif target_hand == "right":
            p_comp = bc.traj.punch_completed_right
            p_h_comp = bc.traj.punch_half_completed_right
        else:
            p_comp = False
            p_h_comp = False

        p_comp = {"punch_completed": p_comp,
                  "punch_half_completed": p_h_comp
                  }
        return json.dumps(p_comp, default=serialize)

    else:
        print("Problem")


@app.route('/compute_punch_metrics/<target_hand>', methods=['GET'])
def compute_punch_metrics(target_hand):
    if request.method == 'GET':
        pm = bc.get_punch_metrics(target_hand)
        return json.dumps(pm, default=serialize)

    else:
        print("Problem")


@app.route('/eval_values/<action>', methods=['GET'])
def evaluation_values(action):
    if request.method == 'GET':
        if action == "record":
            pm = {}
            bc.eval_values(record=True)
            pm["recorded"] = True
            return json.dumps(pm, default=serialize)
        elif action == "save":

            pm = {}
            # global n_punches_eval
            global eval_csv_name
            # bc.eval_values(save=True, save_path=eval_save_path.format(n_punches=str(n_punches_eval)))
            bc.eval_values(save=True, save_path=eval_save_path.format(eval_csv_name=eval_csv_name))
            # n_punches_eval = 0
            eval_csv_name = ""

            pm["saved"] = True
            return json.dumps(pm, default=serialize)

    else:
        print("Problem")


@app.route('/fetch_zp', methods=['GET', 'POST'])
def get_zero_posture():
    """
    This function is associated with a flask route to send the initial posture to be animated on Unity. Executed at the
    beginning of the visualization cycle.

    @return: str, json data of the posture is sent as string to Unity
    """
    if request.method == 'GET':
        posture = controller_to_posture()
        bc.reset([0, 0, 0], 0.0, [0, 0, 1])
        return json.dumps(posture, default=serialize)

    else:
        print("Problem")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
