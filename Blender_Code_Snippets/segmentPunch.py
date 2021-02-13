import bpy # base blender python api
from mathutils import * # imports Vector, Matrix, Quaternion etc. 
import math # standard python math functionality
import pickle
import numpy as np

def load_bvh(path):
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 2
    bpy.ops.import_anim.bvh(filepath=path, 
        axis_forward='X', axis_up='Z', update_scene_fps=True,
        update_scene_duration=True, rotate_mode="YXZ")
    
    return (bpy.context.scene.objects[-1], 
            bpy.context.scene.frame_end)


def get_punches(o, threshold_mag_dist = .05, spine_top = 'Spine3',forearm_l = 'LeftForeArm', forearm_r = 'RightForeArm'):
    
    sp = o.pose.bones[spine_top]
    fl = o.pose.bones[forearm_l]
    fr = o.pose.bones[forearm_r]

    pos_sp_mag = (o.matrix_world @ sp.head).length    
    pos_l_mag = (o.matrix_world @ fl.head).length
    pos_r_mag = (o.matrix_world @ fr.head).length
    
    left_mag_dist = pos_l_mag - pos_sp_mag
    right_mag_dist = pos_r_mag - pos_sp_mag
    
    mag_dist_l.append(left_mag_dist)
    mag_dist_r.append(right_mag_dist)
    
    if left_mag_dist > right_mag_dist and left_mag_dist > threshold_mag_dist:
        res.append([forearm_l, pos_l_mag])
        return (forearm_l, pos_l_mag)
    
    elif right_mag_dist > left_mag_dist and right_mag_dist > threshold_mag_dist:
        res.append([forearm_r, pos_r_mag])
        return (forearm_r, pos_r_mag)
        
    else:
        res.append([None, None])
        return (None, None)

def printListElementOneByOne(L):
    
    for i in L:
        print(i)

##################################################################################################################
print("Start.")

if len(bpy.context.scene.objects) == 0:
    #o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/Data/MocapBoxing/processed/Scene1_NeutralPos.bvh")
    o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/Data/MocapBoxing/processed/Scene5_Punches.bvh")

else:
    o = bpy.context.scene.objects[0] # currently selected object

mag_dist_l = []
mag_dist_r = []
res = []

print('Bones: ')
for i in list(o.pose.bones):
    print(i)

print('Start: ')
frame_start = bpy.context.scene.frame_start
frame_end = bpy.context.scene.frame_end
for f in range(frame_start, frame_end):
    bpy.context.scene.frame_set(f)
    # bpy.context.scene.update() # bpy 2.7x
    bpy.context.view_layer.update() # bpy 2.8x: 
    punch_info = get_punches(o)
    print('Frame: ', f, 'Hand: ', punch_info[0], 'Dist_Mag: ', punch_info[1])
    
print('Done with loop.')

print('##################################################################################################################')
printListElementOneByOne(mag_dist_l)

print('##################################################################################################################')
printListElementOneByOne(mag_dist_r)
    
print('Done.')

with open('/home/chirag/Documents/Uni/Thesis/Blender_Save_Vars/Scene5_Punches.pkl', 'wb') as f:
    pickle.dump([res, mag_dist_l, mag_dist_r], f)