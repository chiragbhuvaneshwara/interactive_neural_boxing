import bpy # base blender python api
from mathutils import * # imports Vector, Matrix, Quaternion etc. 
import math # standard python math functionality

bpy.ops.object.select_all(action='DESELECT')

# https://wiki.blender.org/wiki/Reference/Release_Notes/2.80/Python_API/Scene_and_Object_API
# Blender 2.8x
bpy.data.objects['Camera'].select_set(True)    
bpy.data.objects['Cube'].select_set(True)
bpy.data.objects['Light'].select_set(True)

bpy.ops.object.delete() 
