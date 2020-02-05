import bpy # base blender python api
from mathutils import * # imports Vector, Matrix, Quaternion etc. 
import math # standard python math functionality


o = bpy.context.object
start_loc = Vector(o.pose.bones[0].location)

print(start_loc)


print("Begining to move the object")
for f in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
    
    if (f %50 == 0):
        print(f)
    bpy.context.scene.frame_set(f)
    o.pose.bones[0].location -= start_loc
    o.pose.bones[0].keyframe_insert(data_path="location")
    
print("Done moving.")
