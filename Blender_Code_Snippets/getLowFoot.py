"""
Get the foot contact information. 
Which foot is touching the floor.

Janis implementation: Use distance of feet w.r.t world coordinate system to tell which foot is ahead

Chirag implementation: Same as Janis and also use the speed of the foot to compute foot contact.
                

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


# find the lowest foot
def get_low_foot(o, foot_l = "LeftFoot", foot_r = "RightFoot"):
    fl = o.pose.bones[foot_l]
    fr = o.pose.bones[foot_r]
    
    # mid-position of foot joints in global coordinate system
    pos_l = o.matrix_world @ ((fl.tail + fl.head) / 2)
    pos_r = o.matrix_world @ ((fr.tail + fr.head) / 2)
    if pos_l.z > pos_r.z:
        return (foot_r, pos_r)
    else:
        return(foot_l, pos_l)
    
def get_low_foot_new(o, vel, prev_frame_pos, foot_l = "LeftFoot", foot_r = "RightFoot"):
    fl = o.pose.bones[foot_l]
    fr = o.pose.bones[foot_r]
    
    # mid-position of foot joints in global coordinate system
    pos_l = o.matrix_world @ ((fl.tail + fl.head) / 2)
    pos_r = o.matrix_world @ ((fr.tail + fr.head) / 2)
    
    curr_vel_left = (pos_l.length - prev_frame_pos['left'].length) / (1)
    curr_vel_right = (pos_r.length - prev_frame_pos['right'].length) / (1)
    #curr_vel = [np.array(curr_vel_left), np.array(curr_vel_right)]
    curr_vel = [curr_vel_left, curr_vel_right]
    vel.append(curr_vel)
    
    prev_frame_pos = {'left': pos_l, 'right': pos_r}    
    
    # How to use velocity? Will have to mark some as left, right and None.
    # if right foot higher and right foot speed greater
    if pos_l.z < pos_r.z and curr_vel_left < curr_vel_right:
        return (foot_r, pos_r, prev_frame_pos)
    else:
        return(foot_l, pos_l, prev_frame_pos)
    
    
        
def get_init_pos(o, foot_l = "LeftFoot", foot_r = "RightFoot"):
        
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    # bpy.context.scene.update() # bpy 2.7x
    bpy.context.view_layer.update() # bpy 2.8x: 

    fl = o.pose.bones['LeftFoot']
    fr = o.pose.bones['RightFoot']

    # mid-position of foot joints in global coordinate system
    pos_l = o.matrix_world @ ((fl.tail + fl.head) /2)
    pos_r = o.matrix_world @ ((fr.tail + fr.head) / 2)

    return pos_l, pos_r

##################################################################################################################
print("Start.")

if len(bpy.context.scene.objects) == 0:
    o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/RawData/MocapBoxing/processed/Scene3_Motion.bvh")

else:
    o = bpy.context.scene.objects[0] # currently selected object

vel = []
low_foot_list = []    
init_pos = get_init_pos(o)
prev_frame_pos = {'left': init_pos[0], 'right': init_pos[1]}

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
    print('Frame: ', f)
    low_foot_info = get_low_foot_new(o, vel, prev_frame_pos)
    low_foot = low_foot_info[:2]
    low_foot_list.append([np.array(low_foot[0]), np.array(low_foot[1])])
    
    prev_frame_pos = low_foot_info[2]
    #vel = low_foot_info[3]
        
    print(low_foot)

print('Done with loop.')

#for i in range(len(vel)):
    #vel[i] = np.array(vel[i])
    #print(i, ':', vel[i])
    
with open('/home/chirag/Documents/Uni/Thesis/Blender_Save_Vars/Scene1_FootContact.pkl', 'wb') as f:
    pickle.dump([vel, low_foot_list], f)
    
print('Done.')
