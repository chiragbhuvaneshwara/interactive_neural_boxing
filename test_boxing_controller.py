from src.controlers.boxing_controller import BoxingController
from src.nn.fc_models.mann_tf import *
from src.nn.fc_models.mann_tf import MANN as MANNTF
from src.controlers.character import *
from simple_plotter import simple_matplotlib_plotter
from pathlib import Path, PureWindowsPath
import numpy as np

target_file = PureWindowsPath(
    r"C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\trained_models\mann\epoch_9.json")
target_file = Path(target_file)
mann = MANNTF.load(target_file)
args_dataset = "./data/boxing_config3.json"

with open(args_dataset) as f:
    config_store = json.load(f)

zero_posture = config_store["zero_posture"]

def find_joint_index(name):
    names = []
    for j in zero_posture:
        names.append(j["name"])
    names

    for j in zero_posture:
        if j["name"] == name:
            return j["index"]
# e = find_joint_index('RightFoot')
# f = find_joint_index('RightLeg')
# g = find_joint_index('LeftFoot')
# h = find_joint_index('LeftLeg')
#
# a = find_joint_index('RightShoulder')
# b = find_joint_index('RightHand')
# c = find_joint_index('LeftShoulder')
# d = find_joint_index('LeftHand')

mann.start_tf()
bc = BoxingController(mann, config_store)
#c = Character(config_store)
test_dataset = args_dataset.split('_')[0] + '_test.npz'

data = np.load(test_dataset)
X = data["Xun"]
Y = data["Yun"]

punch_targets = X[:, 2: 8]

my_plotter = simple_matplotlib_plotter.Plotter()

target = [3.94847062, 0, 27.48042372, 0.0, 0.0, 0.0]

poses = []
for f in range(100):
    bc.pre_render(target)
    poses.append(np.array(bc.char.joint_positions))
    # bc.post_render()

poses = np.array(poses)
my_plotter.animated(poses)
