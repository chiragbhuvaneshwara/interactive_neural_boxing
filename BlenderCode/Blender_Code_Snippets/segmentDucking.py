"""
Get the foot contact information. 
Which foot is touching the floor.

Janis implementation: Use distance of feet w.r.t world coordinate system to tell which foot is ahead

Chirag implementation: Same as Janis and also use the speed of the foot to compute foot contact.

To be implemented: 1. Find a good threshold for speed below which foot can be considered to be resting
                   2. Use speed and distance to provide foot contact info.
                   3. Mark foot contact info in columns => |frame|left|right|
                                                           |1    |  0 |  1  |                    

"""

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


def get_ducks(o, threshold = .05, spine_top = 'Spine3', hip = 'Hips'):
    
    sp = o.pose.bones[spine_top]
    hp = o.pose.bones[hip]

    pos_sp = o.matrix_world @ sp.head   
    pos_hp = o.matrix_world @ hp.head
    
    dist = pos_sp.x - pos_hp.x
    
    sp_dist.append(pos_sp.x)
    hp_dist.append(pos_hp.x)
    
    
    glb_dist.append(dist)
    
    if dist > threshold:
        res.append('left')
        return 'left'
        
    elif dist < -threshold:
        res.append('right')
        return 'right'
    else:
        res.append('None')
        return 'None'

def printListElementOneByOne(L):
    
    for i in L:
        print(i)

##################################################################################################################
print("Start.")

if len(bpy.context.scene.objects) == 0:
    #o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/RawData/MocapBoxing/processed/Scene1_NeutralPos.bvh")
    #o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/RawData/MocapBoxing/processed/Scene5_Punches.bvh")
    o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/RawData/MocapBoxing/processed/Scene6_Ducking.bvh")
    
else:
    o = bpy.context.scene.objects[0] # currently selected object

sp_dist = []
hp_dist = []
glb_dist = []
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
    duck_info = get_ducks(o)
    print('Frame: ', f, 'Dir: ', duck_info)
    
print('Done with loop.')

print('##################################################################################################################')
printListElementOneByOne(sp_dist)

print('##################################################################################################################')
printListElementOneByOne(hp_dist)
    
print('Done.')

with open('/home/chirag/Documents/Uni/Thesis/Blender_Save_Vars/Ducks.pkl', 'wb') as f:
    pickle.dump([res, sp_dist, hp_dist, glb_dist], f)