import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 10, 4

gating_out_base = "miscellaneous/Gating_Outs"
# file = "left_punch.json"
file = "right_punch.json"
# file = "forward_stepping.json"
# file = "reverse_stepping.json"

f = open(os.path.join(gating_out_base, file), "r")

# Reading from file
data = json.loads(f.read())
data = np.array(data)

print(data)
fig, ax = plt.subplots(figsize=(10, 4))

for i in range(data.shape[1]):
    ax.plot(data[:, i])
plt.title(" ".join(file.split(".")[0].split("_")))
plt.ylabel('Gating Preds')
plt.xlabel('Frames')
ax.set_xlim([0, 45])
# ax.set_xlim([0,140])

plt.legend([i for i in range(8)], loc=(1.04, 0))
plt.savefig(os.path.join(gating_out_base, "plots", file.split(".")[0] + ".png"))

# Closing file
f.close()
