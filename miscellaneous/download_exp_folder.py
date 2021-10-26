import argparse
import os
from common.utils import retrieve_name
import subprocess

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-e", "--exp_dir", help="example: learning_rate",
                         type=str, required=True)
args = args_parser.parse_args()
exp_dir = args.exp_dir

# model_id = "fr_1_tr_5_5_ep_300/2021-09-14_10-54-55"
# model_id = "learning_rate/fr_1_tr_5_5_ep_300/2021-09-14_10-54-55"
# epoch_id = int(model_id.split("/")[1].split("_")[-1])
src = "chbh01/train-env-v7:/tf/interactive_neural_boxing/train/models/mann_tf2_v2"
dest = "/Users/chbh01/Documents/OfflineCodebases/UdS_Thesis/AllInIOneVCS/interactive_neural_boxing/" \
       "interactive_neural_boxing/train/models/mann_tf2_v2"
REMOTE_SRC = "/tf/interactive_neural_boxing/train/models/mann_tf2_v2"
EXP_DIR = os.path.join(REMOTE_SRC, exp_dir)


def get_dirs(curr_dir):
    p = subprocess.Popen(("kubectl exec train-env-v7 -- ls " + curr_dir).split(" "), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")
    out = out.split("\n")[:-1]
    return out


exp_dir_folders = get_dirs(EXP_DIR)
exp_dir_folders = get_dirs(EXP_DIR)

for m_id in get_dirs(EXP_DIR):
    for m_time_id in get_dirs(os.path.join(EXP_DIR, m_id)):
        print("####################")
        print(m_time_id)
        model_id = os.path.join(exp_dir, m_id, m_time_id)
        epoch_id = int(model_id.split("/")[1].split("_")[-1]) - 1

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
