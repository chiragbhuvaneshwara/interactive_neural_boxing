import bpy, math
from mathutils import *
import pandas as pd
import matplotlib.pyplot as plt

"""left arm instead of shoulder"""

def SetFrame(f):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()

o = bpy.context.object
left = [o.pose.bones["LeftShoulder"], o.pose.bones["LeftHand"]]
right = [o.pose.bones["RightShoulder"], o.pose.bones["RightHand"]]
arm = left
fs = 850
fe = 1050
#fe = 5600

print('######')
print('######')
print('######')
print('######')

dists = []
for f in range(fs, fe):
    SetFrame(f)
    # calculate hand distance
    handDistance = (arm[1].head - arm[0].head).length
    print(f, handDistance)
    dists.append(handDistance)
  
wrt = dists[38]
  
plt.plot(dists)
plt.savefig(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Blender Code Snippets\saved_vis_data\dists.png')
plt.close()
print('######')

last = 0 
dists = []
for f in range(fs, fe):
    SetFrame(f)
    # calculate hand distance
    handDistance = (arm[1].head - arm[0].head).length
    print(f, wrt - handDistance)
    dists.append(last - handDistance)
    last = handDistance    
plt.plot(dists[1:])
plt.savefig(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Blender Code Snippets\saved_vis_data\diff_pos_dists.png')
plt.close()
print('######', dists[60])

last = 0
dists = []
for f in range(fe, fs, -1):
    SetFrame(f)
    # calculate hand distance
    handDistance = (arm[1].head - arm[0].head).length
    print(f, wrt - handDistance)
    dists.append(last - handDistance)
    last = handDistance    
plt.plot(dists[1:])
#print([i for i in range(len(dists[1:]),0, -10)])
plt.xlim(len(dists[1:]), 0)  # decreasing time
#plt.xticks([i for i in range(len(dists[1:]),0, -10)])
plt.savefig(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-boxing-predictor\Blender Code Snippets\saved_vis_data\diff_neg_dists.png')

plt.close()
print('######', dists[18])