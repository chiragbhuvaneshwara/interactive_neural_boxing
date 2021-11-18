import json
import os

from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_epoch_loss(filename):
    epoch_loss = []
    # for e in summary_iterator("train/models/mann_tf2_v2/experiment_0/learning_rate/fr_1_tr_5_5_ep_150/2021-09-21_06-33-07/logs/train/events.out.tfevents.1632205990.asr-vm-lasvegas-01.5082.67.v2"):
    for e in summary_iterator(filename):
        for v in e.summary.value:
            if v.tag == 'epoch_loss':
                epoch_loss.append(v.simple_value)
                # print(v.simple_value)
    return epoch_loss


# for summary in summary_iterator("train/models/mann_tf2_v2/experiment_0/learning_rate/fr_1_tr_5_5_ep_150/2021-09-21_06-33-07/logs/train/events.out.tfevents.1632205990.asr-vm-lasvegas-01.5082.67.v2"):
#     summary
#     print(summary)
exp_folder = "train/models/mann_tf2_v2/experiment_0/learning_rate"
# exp_folder = "train/models/mann_tf2_v2/experiment_0/num_gating_experts"
# exp_folder = "train/models/mann_tf2_v2/experiment_0/num_hidden_neurons"
for exp in [f for f in os.listdir(exp_folder) if os.path.splitext(f)[1] == ""]:
    curr_exp = os.path.join(exp_folder, exp)

    results = {}
    for timestamp in [f for f in os.listdir(curr_exp) if os.path.splitext(f)[1] == ""]:
        curr_exp_timestamp = os.path.join(curr_exp, timestamp)

        curr_exp_timestamp_config = os.path.join(curr_exp_timestamp, "network_config.json")
        with open(curr_exp_timestamp_config) as json_file:
            config = json.load(json_file)
        if "learning_rate" in exp_folder:
            curr_key = config["learning_rate"]["initial_rate"]
        elif "num_gating_experts" in exp_folder:
            curr_key = config["num_experts"]
        elif "num_hidden_neurons" in exp_folder:
            curr_key = config["num_hidden_neurons"]

        curr_tb = os.path.join(curr_exp, timestamp, "logs", "train")
        curr_tb_file = os.path.join(curr_tb, [f for f in os.listdir(curr_tb) if os.path.splitext(f)[1] == '.v2'][0])
        e_loss = get_epoch_loss(curr_tb_file)
        results[curr_key] = e_loss
        # results[curr_key] = len(e_loss)

if exp_folder == "train/models/mann_tf2_v2/experiment_0/learning_rate":
    del results[0.005]
    del results[0.05]

results = {exp_folder: results}
with open(os.path.join(exp_folder, 'epoch_loss.json'), 'w') as fp:
    json.dump(results, fp, indent=4)
print(results)

data = results[exp_folder]
df = pd.DataFrame.from_dict(data)
print(df)

if "learning_rate" in exp_folder:
    legend_title = "learning rate \u03B7"
    super_title = "Loss plot for tuning learning rate"
elif "num_gating_experts" in exp_folder:
    legend_title = "# gating experts"
    super_title = "Loss plot for tuning number of gating experts"
elif "num_hidden_neurons" in exp_folder:
    legend_title = "# hidden neurons"
    super_title = "Loss plot for tuning number of hidden neurons"

plt.figure(dpi=1200)
loss_sns_plot = sns.lineplot(data=df)
loss_sns_plot.legend_.set_title(legend_title)
fig = loss_sns_plot.get_figure()
fig.axes[0].set_ylim([0.04, 0.125])
# title and labels, setting initial sizes
fig.suptitle(super_title, fontsize="large")
fig.axes[0].set_xlabel('epochs', fontsize="large")
fig.axes[0].set_ylabel('MSE loss', fontsize='large')  # relative to plt.rcParams['font.size']

fig.savefig(os.path.join(exp_folder, "loss.png"), dpi=300)
