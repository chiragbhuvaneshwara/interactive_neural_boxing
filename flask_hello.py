from flask import Flask, request
from src.controlers.boxingControllers.controller_box import BoxingController
import json, os
from src.nn.keras_mods.mann_keras import MANN as MANNTF
from flask import jsonify

frd = 1
# window = 25
window = 15
epochs = 60
controller_in_out_dir = 'src/controlers/boxingControllers/controller_in_out'
frd_win_epochs = 'boxing_fr_' + str(frd) + '_' + str(window) + '_' + str(epochs)
trained_base_path = 'trained_models/mann_tf2/' + frd_win_epochs
target_file = os.path.join(trained_base_path, 'model_weights_std_in_out.zip')
mann_config_path = os.path.join(trained_base_path, 'mann_config.json')
with open(mann_config_path) as json_file:
    mann_config = json.load(json_file)


mann = MANNTF(mann_config)
mann.load_discrete_weights(target_file)
dataset_config = "data/boxing_fr_" + str(frd) + "_" + str(window) + "/config.json"

with open(dataset_config) as f:
    config_store = json.load(f)

bc = BoxingController(mann, config_store)

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/punch_in', methods=['GET', 'POST'])
def post_req():
    print('Here')
    if request.method == 'POST':  # this block is only entered when the form is submitted
        punch_in = request.get_json()
        punch_hand = punch_in["hand"]
        punch_target = punch_in["target_right"] + punch_in["target_left"]
        bc.pre_render(punch_target)
        pose = bc.char.joint_positions
        # pose = [1, 2, 3]
        print(pose)
    return jsonify(pose.tolist())


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
