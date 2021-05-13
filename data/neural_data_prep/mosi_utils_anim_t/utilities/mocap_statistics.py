import glob
from morphablegraphs.animation_data import BVHReader, SkeletonBuilder
from morphablegraphs.animation_data.utils import get_step_length, \
                                            pose_orientation_quat
import numpy as np
import os
from ..motion_model.motion_primitive import MotionPrimitive
import matplotlib.pyplot as plt

class MocapDataStats(object):
    """
    get statistical information e.g. maximum step length, minimum step length, average step lenght... from motion
    capture data
    """

    def __init__(self):
        self.data_folder = None
        self.bvhreaders = []
        self.skeleton = None

    def get_data_folder(self, folder_path):
        """
        Give the folder path of bvh files
        :param folder_path: string, path to motion capture data
        """
        if not folder_path.endswith(os.sep):
            folder_path += os.sep
        self.data_folder = folder_path
        self._load_data()

    def _load_data(self):
        """
        Load bvh files from data folder

        """
        bvh_files = glob.glob(self.data_folder + '*.bvh')
        for item in bvh_files:
            bvh_reader = BVHReader(item)
            self.bvhreaders.append(bvh_reader)
        self.skeleton = SkeletonBuilder().load_from_bvh(bvh_reader)
        # self.calculate_step_len()

    def calculate_step_len(self):
        """
        Calculate step length of files in bvhreaders
        """
        self.mocap_data_stepLen = {}
        for bvhreader in self.bvhreaders:
            step_len = get_step_length(bvhreader.frames)
            self.mocap_data_stepLen[bvhreader.filename] = step_len

    def get_max_step_length(self):
        """
        Calculate the maximum step length in mocap data
        :return max_len: maximum step length in mocap data
        """
        return np.max(list(self.mocap_data_stepLen.values()))

    def get_min_step_length(self):
        """
        Calculate the minimum step length in mocap data
        :return min_len: minimum step length in mocap data
        """
        return np.min(list(self.mocap_data_stepLen.values()))

    def get_average_step_length(self):
        """
        Calculate average step length in mocap data
        :return aver_len: average step length in mocap data
        """
        return np.mean(list(self.mocap_data_stepLen.values()))

    def get_joint_position_for_all_motion(self, joint_name, frame_idx):
        joint_positions = []
        for bvhreader in self.bvhreaders:
            frame = bvhreader.frames[frame_idx]
            pos = self.skeleton.nodes[joint_name].get_global_position_from_euler_frame(frame,
                                                                                       rotation_order=['Xrotation',
                                                                                                       'Yrotation',
                                                                                                       'Zrotation'])
            joint_positions.append(pos)
        return joint_positions


class MotionPrimitiveStats(object):
    def __init__(self, mm_file):
        self.motion_primitive_model = MotionPrimitive(mm_file)

    def evaluate_sample_orientation(self, N):
        orientation_vecs = []
        plt.figure()
        for i in range(N):
            motion_sample = self.motion_primitive_model.sample()
            quat_frames = motion_sample.get_motion_vector()
            pose_orientation = pose_orientation_quat(quat_frames[0])
            print(pose_orientation)
            plt.plot([0, pose_orientation[0]], [0, pose_orientation[1]], 'r')
            orientation_vecs.append(pose_orientation)
        orientation_vecs = np.asarray(orientation_vecs)
        # plt.figure()
        # plt.plot(orientation_vecs[:,0], orientation_vecs[:,1], 'r.')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()

    def evaluate_touch_ground_joint(self, N):
        left_toe_first_pos = []
        left_toe_last_pos = []
        right_toe_first_pos = []
        left_toe_last_pos = []
        left_toe_joint_name = 'Bip01_L_Toe0'
        right_toe_first_pos = 'Bip01_R_Toe0'

