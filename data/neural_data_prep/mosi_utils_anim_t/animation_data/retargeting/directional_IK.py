
import copy
import numpy as np
import os
from ..skeleton_builder import SkeletonBuilder
from ..retargeting.directional_constraints_retargeting import align_ref_frame, retarget_motion
from ..body_plane import BodyPlane
from ..bvh import BVHWriter


class DirectionalIK(object):

    def __init__(self, target_bvhreader):
        self.target_bvhreader = target_bvhreader
        self.skeleton = SkeletonBuilder().load_from_bvh(self.target_bvhreader)
        self.skeleton_scale_factor = 1.0

    def set_scale_factor(self, scale):
        self.skeleton_scale_factor = scale
    
    def __call__(self, points, points_des, body_plane_joints):
        """
        
        Arguments:
            points {numpy.array 3d} -- point cloud motion data
            points_des {[type]} -- the list of joint sequence. It is used to compute bone direction
            e.g. 
                h6M_SKELETON = [
                    {'name': 'Hips', 'parent': None, 'index': 0},
                    {'name': 'HipRight', 'parent': 'Hips', 'index': 1},
                    {'name': 'KneeRight', 'parent': 'HipRight', 'index': 2},
                    {'name': 'FootRight', 'parent': 'KneeRight', 'index': 3},
                    {'name': 'ToeBaseRight', 'parent': 'FootRight', 'index': 4},
                    {'name': 'Site1', 'parent': 'ToeBaseRight', 'index': 5},
                    {'name': 'HipLeft', 'parent': 'Hips', 'index': 6},
                    {'name': 'KneeLeft', 'parent': 'HipLeft', 'index': 7},
                    {'name': 'FootLeft', 'parent': 'KneeLeft', 'index': 8},
                    {'name': 'ToeBaseLeft', 'parent': 'FootLeft', 'index': 9},
                    {'name': 'Site2', 'parent': 'ToeBaseLeft', 'index': 10},
                    {'name': 'Spine1', 'parent': 'Hips', 'index': 11},
                    {'name': 'Spine2', 'parent': 'Spine1', 'index': 12},
                    {'name': 'Neck', 'parent': 'Spine2', 'index': 13},
                    {'name': 'Head', 'parent': 'Neck', 'index': 14},
                    {'name': 'Site3', 'parent': 'Head', 'index': 15},
                    {'name': 'ShoulderLeft', 'parent': 'Neck', 'index': 16},
                    {'name': 'ElbowLeft', 'parent': 'ShoulderLeft', 'index': 17},
                    {'name': 'WristLeft', 'parent': 'ElbowLeft', 'index': 18},
                    {'name': 'HandLeft', 'parent': 'WristLeft', 'index': 19},
                    {'name': 'HandThumbLeft', 'parent': 'HandLeft', 'index': 20},
                    {'name': 'Site4', 'parent': 'HandThumbLeft', 'index': 21},
                    {'name': 'WristEndLeft', 'parent': 'HandLeft', 'index': 22},
                    {'name': 'Site5', 'parent': 'WristEndLeft', 'index': 23},
                    {'name': 'ShoulderRight', 'parent': 'Neck', 'index': 24},
                    {'name': 'ElbowRight', 'parent': 'ShoulderRight', 'index': 25},
                    {'name': 'WristRight', 'parent': 'ElbowRight', 'index': 26},
                    {'name': 'HandRight', 'parent': 'WristRight', 'index': 27},
                    {'name': 'HandThumbRight', 'parent': 'HandRight', 'index': 28},
                    {'name': 'Site6', 'parent': 'HandThumbRight', 'index': 29},
                    {'name': 'WristEndRight', 'parent': 'HandRight', 'index': 30},
                    {'name': 'Site7', 'parent': 'WristEndRight', 'index': 31},
                ]
        """
        n_frames = len(points)
        directional_constraints = self._create_directional_constraints(points, points_des, body_plane_joints)
        self.out_frames = []
        root_index = next(item['index'] for item in points_des if item['name'] == self.skeleton.root)
        for i in range(n_frames):
            pose_dir = directional_constraints[i]['pose_dir']
            if i == 0:
                new_frame = self.target_bvhreader.frames[0]  ## take the first frame as reference frame
                ref_frame = align_ref_frame(new_frame, pose_dir, self.skeleton, body_plane_joints)
                ## set global position
                ref_frame[0] = points[0, root_index, 0] * self.skeleton_scale_factor
                ref_frame[2] = points[0, root_index, 2] * self.skeleton_scale_factor
            else:
                new_frame = copy.deepcopy(self.out_frames[i-1])
                ref_frame = align_ref_frame(new_frame, pose_dir, self.skeleton, body_plane_joints)
                ref_frame[:3] = (points[i, root_index] - points[i-1, root_index]) * self.skeleton_scale_factor + self.out_frames[i-1][:3]
            retarget_motion(self.skeleton.root, directional_constraints[i], self.skeleton, ref_frame)
            self.out_frames.append(ref_frame)
    
    def save_as_bvh(self, save_path):
        """save retarget motion as bvh file
        
        Arguments:
            save_path {str}
        """
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(save_path)
        BVHWriter(save_path, self.skeleton, self.out_frames, self.skeleton.frame_time, is_quaternion=False)

    def _create_directional_constraints(self, points, points_des, body_plane_joints):
        """compute directional constraints from global positions
        
        Arguments:
            points {numpy.array3d} -- n_frames * n_joints * 3
        """
        n_frames, n_joints, _ = points.shape
        constraints = []
        if isinstance(points_des, list):
            for i in range(n_frames):
                frame_constraints = {}
                if body_plane_joints is not None:
                    torso_points = []
                    for joint in body_plane_joints:
                        joint_index = next(item['index'] for item in points_des if item['name'] == joint)
                        torso_points.append(points[i, joint_index])
                    body_plane = BodyPlane(torso_points)
                    dir_vec = np.array([body_plane.normal_vector[0], body_plane.normal_vector[2]])
                    frame_constraints['pose_dir'] = dir_vec / np.linalg.norm(dir_vec)
                else:
                    frame_constraints['pose_dir'] = None
                for joint_des in points_des:
                    # joint_parent = next(item['parent'] for item in points_des if item['name'] == joint[''])
                    if joint_des['parent'] is not None:
                        parent_index = next(item['index'] for item in points_des if item['name'] == joint_des['parent'])
                        joint_dir = points[i, joint_des['index']] - points[i, parent_index]
                        assert np.linalg.norm(joint_dir) != 0  ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                        if joint_des['parent'] in frame_constraints.keys():
                            frame_constraints[joint_des['parent']][joint_des['name']] = joint_dir / np.linalg.norm(joint_dir)
                        else:
                            frame_constraints[joint_des['parent']] = {joint_des['name']: joint_dir / np.linalg.norm(joint_dir)}
                constraints.append(frame_constraints)
        else:
            raise IOError("Unknown data type! ")
        return constraints