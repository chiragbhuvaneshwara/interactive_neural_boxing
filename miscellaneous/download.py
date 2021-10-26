import argparse
import os
from common.utils import retrieve_name

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-m", "--model_id", help="example: learning_rate/fr_1_tr_5_5_ep_300/2021-09-14_10-54-55",
                         type=str, required=True)
args = args_parser.parse_args()
model_id = args.model_id

# model_id = "fr_1_tr_5_5_ep_300/2021-09-14_10-54-55"
# model_id = "learning_rate/fr_1_tr_5_5_ep_300/2021-09-14_10-54-55"
try:
    epoch_id = int(model_id.split("/")[0].split("_")[-1]) - 1
except:
    epoch_id = int(model_id.split("/")[1].split("_")[-1]) - 1
src = "chbh01/train-env-v7:/tf/interactive_neural_boxing/train/models/mann_tf2_v2"
dest = "/Users/chbh01/Documents/OfflineCodebases/UdS_Thesis/AllInIOneVCS/interactive_neural_boxing/" \
       "interactive_neural_boxing/train/models/mann_tf2_v2"

epoch_cmd = "kubectl cp {src} {dest}".format(
    src=os.path.join(src, model_id, "epochs", "epoch_{epoch_id}".format(epoch_id=epoch_id)),
    dest=os.path.join(dest, model_id, "epochs", "epoch_{epoch_id}".format(epoch_id=epoch_id)))
config_cmd = "kubectl cp {src} {dest}".format(
    src=os.path.join(src, model_id, "network_config.json"),
    dest=os.path.join(dest, model_id, "network_config.json"))
logs_cmd = "kubectl cp {src} {dest}".format(
    src=os.path.join(src, model_id, "logs"),
    dest=os.path.join(dest, model_id, "logs"))

cmds = [epoch_cmd, config_cmd, logs_cmd]
cmd_names = list(map(retrieve_name, cmds))
for cmd_name, cmd in zip(cmd_names, cmds):
    os.system(cmd)
    print(cmd_name, "finished")

# python -m miscellaneous.download -m fr_1_tr_5_5_ep_300/2021-09-14_10-54-55
