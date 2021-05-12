from mosi_utils_anim.animation_data import BVHReader, Skeleton, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames
from scipy.misc import imresize
import numpy as np

class MotionImage():
    
    def __init__(self, bvh_file_path, img_height=224, interpolation='bicubic', normalization='local'):
        self.bvh_file_path = bvh_file_path
        self.img_height = img_height
        self.interpolation = interpolation
        self.normalization = normalization
        self.animated_joints = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]

    def load_motion(self):
        bvhreader = BVHReader(self.bvh_file_path)
        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        euler_frames = bvhreader.frames
        if self.normalization == 'local':
            euler_frames[:, :6] = 0
            
        cart_frames = convert_euler_frames_to_cartesian_frames(skeleton, euler_frames, self.animated_joints)
        self.cart_motion = cart_frames

    def map_motion_to_image(self):
        motion = np.asarray(self.cart_motion)
        n_frames = motion.shape[0]
        print(motion.shape)

        x_arr = motion[:, :, 0].T
        y_arr = motion[:, :, 1].T
        z_arr = motion[:, :, 2].T
        print('here', x_arr.shape, y_arr.shape, z_arr.shape)

        r = np.multiply((x_arr-x_arr.min()), 255.0/(x_arr.max()-x_arr.min())).tolist()
        g = np.multiply((y_arr-y_arr.min()), 255.0/(y_arr.max()-y_arr.min())).tolist()
        b = np.multiply((z_arr-z_arr.min()), 255.0/(z_arr.max()-z_arr.min())).tolist()

        r_arr = np.array([l for l in r for i in range(1)])
        g_arr = np.array([l for l in g for i in range(1)])
        b_arr = np.array([l for l in b for i in range(1)])

        rgb_arr = np.dstack((r_arr, g_arr, b_arr))
        rgb_motion = imresize(rgb_arr, (self.img_height, n_frames), interp=self.interpolation, mode=None)
        self.rgb_motion = rgb_motion

