# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:56:45 2015

@author: herrmann
"""
import os
import glob
import numpy as np
import copy
from morphablegraphs.animation_data import BVHReader, MotionVector, SkeletonBuilder
from morphablegraphs.external.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix, quaternion_multiply


def flip_coordinate_system(q):
    """
    as far as i understand it is assumed that we are already in a different coordinate system
    so we first flip the coordinate system to the original coordinate system to do the original transformation
    then flip coordinate system again to go back to the flipped coordinate system
    this results in the original transformation being applied in the flipped transformation
    http://www.ch.ic.ac.uk/harrison/Teaching/L4_Symmetry.pdf
    http://gamedev.stackexchange.com/questions/27003/flip-rotation-matrix
    https://www.khanacademy.org/math/linear-algebra/alternate_bases/change_of_basis/v/lin-alg-alternate-basis-tranformation-matrix-example
    """
    conv_m = np.array([[-1, 0, 0,  0],
                         [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0,  0, 1]])


    m = quaternion_matrix(q)
    new_m = np.dot(conv_m, np.dot(m, conv_m))
    return quaternion_from_matrix(new_m)


def swap_parameters(frame, node_names, mirror_map):
    # mirror joints
    temp = copy.deepcopy(frame[:])
    for node_name in node_names:
        if node_name in mirror_map.keys():
            target_node_name = mirror_map[node_name]
            # print("mirror", node_name, target_node_name)
            src = node_names.index(node_name) * 4 + 3
            dst = node_names.index(target_node_name) * 4 + 3
            frame[dst:dst + 4] = temp[src:src + 4]
    return frame



def flip_root_coordinate_system(q1):
    """
    http://www.gamedev.sk/mirroring-animations
    http://www.gamedev.net/topic/599824-mirroring-a-quaternion-against-the-yz-plane/
    """

    conv_m = np.array([[-1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])


    m = quaternion_matrix(q1)
    new_m = np.dot(conv_m, np.dot(m, conv_m))
    q2 = quaternion_from_matrix(new_m)
    flip_q = quaternion_from_euler(*np.radians([0, 0, 180]))
    q2 = quaternion_multiply(flip_q, q2)
    return q2

def flip_pelvis_coordinate_system(q):
    """
    http://www.gamedev.sk/mirroring-animations
    http://www.gamedev.net/topic/599824-mirroring-a-quaternion-against-the-yz-plane/
    """

    conv_m = np.array([[-1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])


    m = quaternion_matrix(q)
    new_m = np.dot(conv_m, np.dot(m, conv_m))
    q = quaternion_from_matrix(new_m)
    flip_q = quaternion_from_euler(*np.radians([0, 0, 180]))
    q = quaternion_multiply(flip_q, q)
    return q

def mirror_animation(node_names,frames,mirror_map):
    """
    http://www.gamedev.sk/mirroring-animations
    http://stackoverflow.com/questions/1263072/changing-a-matrix-from-right-handed-to-left-handed-coordinate-system
    """
    new_frames = []
    temp = frames[:]
    for frame in temp:
        new_frame = frame[:]
        #handle root separately
        new_frame[:3] =[-new_frame[0],new_frame[1],new_frame[2]]

        # bring rotation into different coordinate system
        for idx, node_name in enumerate(node_names):
            o = idx * 4 + 3
            q = copy.copy(new_frame[o:o + 4])
            if node_name == "pelvis":
                q = flip_pelvis_coordinate_system(q)
            elif node_name == "Root":
                q = flip_root_coordinate_system(q)
            else:
                q = flip_coordinate_system(q)

            new_frame[o:o+4] = q


        new_frame = swap_parameters(new_frame, node_names, mirror_map)

        new_frames.append(new_frame)
    return new_frames


mirror_map = {"clavicle_l": "clavicle_r",
              "upperarm_l": "upperarm_r",
              "lowerarm_l": "lowerarm_r",
              "hand_l": "hand_r",
              "thigh_l": "thigh_r",
              "calf_l": "calf_r",
              "foot_l": "foot_r"
              }
keys = list(mirror_map.keys())
for k in keys:
    mirror_map[mirror_map[k]] = k


def main():
    data_path = r"E:\projects\model_data\hybrit\mirroring\*.bvh"
    out_path = r"E:\projects\model_data\hybrit\mirroring\out"
    file_list = list(glob.glob(data_path))
    bvh_reader = BVHReader(file_list[0])
    joints = list(bvh_reader.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints=joints)

    for filename in file_list:
        output_filename = out_path + os.sep + filename.split(os.sep)[-1][:-4] + "_mirrored.bvh"
        print("write", filename, "to", output_filename)
        mv = MotionVector()
        bvh_reader = BVHReader(filename)
        mv.from_bvh_reader(bvh_reader, False)
        new_frames = mirror_animation(joints, mv.frames, mirror_map)
        mv.frames = new_frames
        mv.export(skeleton, output_filename)


if __name__ == "__main__":
    main()

