import bpy # base blender python api
from mathutils import * # imports Vector, Matrix, Quaternion etc. 
import math # standard python math functionality

def load_bvh(path):
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 2
    bpy.ops.import_anim.bvh(filepath=path, 
        axis_forward='X', axis_up='-Y', update_scene_fps=True,
        update_scene_duration=True, rotate_mode="YXZ")
    
    return (bpy.context.scene.objects[-1], 
            bpy.context.scene.frame_end)


print("Start.")
o, b = load_bvh("/home/chirag/Documents/Uni/Thesis/Data/MocapBoxing/axis_neuron_processed/4_Stepping_AxisNeuronProcessed_Char00.bvh")
