__author__ = 'erhe01'

from datetime import datetime
import os
import numpy as np
from .utils import align_frames,transform_euler_frames, convert_euler_frames_to_quaternion_frames
from .motion_concatenation import align_and_concatenate_frames, smooth_root_positions#, align_frames_and_fix_feet
from .constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER
from .bvh import BVHWriter
import imp
from ..external.transformations import quaternion_inverse, quaternion_multiply, quaternion_slerp



def get_quaternion_delta(a, b):
    return quaternion_multiply(quaternion_inverse(b), a)


def add_frames(skeleton, a, b):
    """ returns c = a + b"""
    #print("add frames", len(a), len(b))
    c = np.zeros(len(a))
    c[:3] = a[:3] + b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx * 4 + 3
        q_a = a[o:o + 4]
        q_b = b[o:o + 4]
        #print(q_a,q_b)
        q_prod = quaternion_multiply(q_a, q_b)
        c[o:o + 4] = q_prod / np.linalg.norm(q_prod)
    return c


def substract_frames(skeleton, a, b):
    """ returns c = a - b"""
    c = np.zeros(len(a))
    c[:3] = a[:3] - b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx*4 + 3
        q_a = a[o:o+4]
        q_b = b[o:o+4]
        q_delta = get_quaternion_delta(q_a, q_b)
        c[o:o+4] = q_delta / np.linalg.norm(q_delta)
    return c

class MotionVector(object):
    """
    Contains quaternion frames,
    """
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        self.n_frames = 0
        self._prev_n_frames = 0
        self.frames = None
        self.start_pose = None
        self.rotation_type = rotation_type
        self.apply_spatial_smoothing = False
        self.apply_foot_alignment = False
        self.smoothing_window = 0
        self.spatial_smoothing_method = "smoothing"
        self.frame_time = 1.0/30.0
        self.skeleton = skeleton

        if algorithm_config is not None:
            settings = algorithm_config["smoothing_settings"]
            self.apply_spatial_smoothing = settings["spatial_smoothing"]
            self.smoothing_window = settings["spatial_smoothing_window"]

            if "spatial_smoothing_method" in settings:
                self.spatial_smoothing_method = settings["spatial_smoothing_method"]
            if "apply_foot_alignment" in settings:
                self.apply_foot_alignment = settings["apply_foot_alignment"]

    def from_bvh_reader(self, bvh_reader, filter_joints=True, animated_joints=None):
        if self.rotation_type == ROTATION_TYPE_QUATERNION:
            self.frames = np.array(convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames, filter_joints, animated_joints))
        elif self.rotation_type == ROTATION_TYPE_EULER:
            self.frames = bvh_reader.frames
        self.n_frames = len(self.frames)
        self._prev_n_frames = 0
        self.frame_time = bvh_reader.frame_time

    def get_relative_frames(self):
        relative_frames = []
        for idx in range(1, len(self.frames)):
            delta_frame = np.zeros(len(self.frames[0]))
            delta_frame[7:] = self.frames[idx][7:]
            delta_frame[:3] = self.frames[idx][:3] - self.frames[idx-1][:3]
            currentq = self.frames[idx][3:7] / np.linalg.norm(self.frames[idx][3:7])
            prevq = self.frames[idx-1][3:7] / np.linalg.norm(self.frames[idx-1][3:7])
            delta_q = quaternion_multiply(quaternion_inverse(prevq), currentq)
            delta_frame[3:7] = delta_q
            #print(idx, self.frames[idx][:3], self.frames[idx - 1][:3], delta_frame[:3], delta_frame[3:7])

            relative_frames.append(delta_frame)
        return relative_frames

    def append_frames_generic(self, new_frames):
        """Align quaternion frames to previous frames

        Parameters
        ----------
        * new_frames: list
            A list of frames with the same rotation format type as the motion vector
        """
        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        self.frames = align_and_concatenate_frames(self.skeleton, self.skeleton.aligning_root_node, new_frames, self.frames, self.start_pose,
                                                   smoothing_window=smoothing_window, blending_method=self.spatial_smoothing_method)

        self.n_frames = len(self.frames)
        self._prev_n_frames = self.n_frames


    def append_frames_using_forward_blending(self, new_frames):
        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        from . import motion_concatenation
        imp.reload(motion_concatenation)
        ik_chains = self.skeleton.skeleton_model["ik_chains"]
        self.frames = motion_concatenation.align_frames_using_forward_blending(self.skeleton, self.skeleton.aligning_root_node, new_frames,
                                                                               self.frames, self._prev_n_frames, self.start_pose,
                                                                               ik_chains, smoothing_window)
        self._prev_n_frames = self.n_frames
        self.n_frames = len(self.frames)

    def append_frames(self, new_frames, plant_foot=None):
        if self.apply_foot_alignment and self.skeleton.skeleton_model is not None:
            self.append_frames_using_forward_blending(new_frames)
        else:
            self.append_frames_generic(new_frames)

    def export(self, skeleton, output_filename, add_time_stamp=False):
        bvh_writer = BVHWriter(None, skeleton, self.frames, skeleton.frame_time, True)
        if add_time_stamp:
            output_filename = output_filename + "_" + \
                       str(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
        elif output_filename != "":
            if not output_filename.endswith("bvh"):
                output_filename = output_filename + ".bvh"
        else:
            output_filename = "output.bvh"
        bvh_writer.write(output_filename)

    def reduce_frames(self, n_frames):
        if n_frames == 0:
            self.frames = None
            self.n_frames = 0
            self._prev_n_frames = self.n_frames
        else:
            self.frames = self.frames[:n_frames]
            self.n_frames = len(self.frames)
            self._prev_n_frames = 0

    def has_frames(self):
        return self.frames is not None

    def clear(self, end_frame=0):
        if end_frame == 0:
            self.frames = None
            self.n_frames = 0
            self._prev_n_frames = 0
        else:
            self.frames = self.frames[:end_frame]
            self.n_frames = len(self.frames)
            self._prev_n_frames = 0

    def translate_root(self, offset):
        for idx in range(self.n_frames):
            self.frames[idx][:3] += offset

    def scale_root(self, scale_factor):
        for idx in range(self.n_frames):
            self.frames[idx][:3] *= scale_factor

    def from_fbx(self, animation, animated_joints=None):
        if animated_joints is None:
            animated_joints = list(animation["curves"].keys())
        self.frame_time = animation["frame_time"]
        print("animated joints", animated_joints)
        root_joint = animated_joints[0]
        self.n_frames = len(animation["curves"][root_joint])
        self.frames = []
        for idx in range(self.n_frames):
            frame = self._create_frame_from_fbx(animation, animated_joints, idx)
            self.frames.append(frame)

    def _create_frame_from_fbx(self, animation, animated_joints, idx):
        n_dims = len(animated_joints) * 4 + 3
        frame = np.zeros(n_dims)
        offset = 3
        root_name = animated_joints[0]
        frame[:3] = animation["curves"][root_name][idx]["local_translation"]
        print("root translation", frame[:3])
        for node_name in animated_joints:
            if node_name in list(animation["curves"].keys()):
                rotation = animation["curves"][node_name][idx]["local_rotation"]
                frame[offset:offset+4] = rotation
            else:
                frame[offset:offset+4] = [1, 0, 0, 0]
            offset += 4

        return frame

    def to_unity_format(self, scale=1.0):
        """ Converts the frames into a custom json format for use in a Unity client"""
        animated_joints = [j for j, n in list(self.skeleton.nodes.items()) if
                           "EndSite" not in j and len(n.children) > 0]  # self.animated_joints
        unity_frames = []

        for node in list(self.skeleton.nodes.values()):
            node.quaternion_index = node.index

        for frame in self.frames:
            unity_frame = self._convert_frame_to_unity_format(frame, animated_joints, scale)
            unity_frames.append(unity_frame)

        result_object = dict()
        result_object["frames"] = unity_frames
        result_object["frameTime"] = self.frame_time
        result_object["jointSequence"] = animated_joints
        return result_object

    def _convert_frame_to_unity_format(self, frame, animated_joints, scale=1.0):
        """ Converts the frame into a custom json format and converts the transformations
            to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """
        unity_frame = {"rotations": [], "rootTranslation": None}
        for node_name in self.skeleton.nodes.keys():
            if node_name in animated_joints:
                node = self.skeleton.nodes[node_name]
                if node_name == self.skeleton.root:
                    t = frame[:3] * scale
                    unity_frame["rootTranslation"] = {"x": -t[0], "y": t[1], "z": t[2]}

                if node_name in self.skeleton.animated_joints:  # use rotation from frame
                    # TODO fix: the animated_joints is ordered differently than the nodes list for the latest model
                    index = self.skeleton.animated_joints.index(node_name)
                    offset = index * 4 + 3
                    r = frame[offset:offset + 4]
                    unity_frame["rotations"].append({"x": -r[1], "y": r[2], "z": r[3], "w": -r[0]})
                else:  # use fixed joint rotation
                    r = node.rotation
                    unity_frame["rotations"].append(
                        {"x": -float(r[1]), "y": float(r[2]), "z": float(r[3]), "w": -float(r[0])})
        return unity_frame

    def from_custom_unity_format(self, data):
        self.frames = []
        for f in data["frames"]:
            t = f["rootTranslation"]
            new_f = [-t["x"], t["y"], t["z"]]
            for q in f["rotations"]:
                new_f.append(-q["w"])
                new_f.append(-q["x"])
                new_f.append(q["y"])
                new_f.append(q["z"])

            self.frames.append(new_f)
        self.frames = np.array(self.frames)
        self.n_frames = len(self.frames)
        self.frame_time = data["frameTime"]

    def apply_low_pass_filter_on_root(self, window):
        self.frames[:, :3] = smooth_root_positions(self.frames[:, :3], window)

    def apply_delta_frame(self, skeleton, delta_frame):
        for f in range(self.n_frames):
            self.frames[f] = add_frames(skeleton, self.frames[f], delta_frame)

    def interpolate(self, start_idx, end_idx, t):
        new_frame = np.zeros(self.frames[0].shape)
        new_frame[:3] = (1-t) * self.frames[start_idx][:3] + t * self.frames[end_idx][:3]
        for i in range(3, new_frame.shape[0], 4):
            start_q = self.frames[start_idx][i:i+4]
            end_q = self.frames[end_idx][i:i+4]
            new_frame[i:i+4] = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        return new_frame