import numpy as np
from .animation_data import SkeletonBuilder, MotionVector, BVHReader, BVHWriter
from .external.transformations import quaternion_from_euler
from .animation_data.retargeting import get_targets_from_motion, ROCKETBOX_TO_GAME_ENGINE_MAP, ADDITIONAL_ROTATION_MAP,GAME_ENGINE_TO_ROCKETBOX_MAP
from .animation_data.retargeting import get_new_frames_from_direction_constraints as get_new_frames_using_quaternion, retarget_from_src_to_target
from .animation_data.retargeting import get_new_euler_frames_from_direction_constraints as get_new_frames_using_euler


def convert_euler_to_quat_frame(skeleton, frame):
    n_dims = len(skeleton.animated_joints) * 4 + 3
    quat_frame = np.zeros(n_dims)
    quat_frame[:3] = frame[:3]
    target_offset = 3
    src_offset = 3
    for idx, node in enumerate(skeleton.animated_joints):
        e = np.radians(frame[src_offset:src_offset + 3])
        q = quaternion_from_euler(*e)
        quat_frame[target_offset:target_offset + 4] = q
        target_offset += 4
        src_offset += 3
    return quat_frame


def create_motion_vector_from_euler_frames(skeleton, euler_frames):
    mv = MotionVector()
    mv.frame_time = skeleton.frame_time
    quat_frames = []
    for e_frame in euler_frames:
        quat_frames.append(convert_euler_to_quat_frame(skeleton, e_frame))
    mv.frames = np.array(quat_frames)
    return mv


def retarget(src_skeleton, src_motion, target_skeleton, inv_joint_map=ROCKETBOX_TO_GAME_ENGINE_MAP, additional_rotation_map=None, frame_range=None, scale_factor=1.0, use_optimization=True, use_euler=False):
    if use_optimization:
        targets = get_targets_from_motion(src_skeleton, src_motion.frames, inv_joint_map, additional_rotation_map=additional_rotation_map)
        if not use_euler:
            new_frames = get_new_frames_using_quaternion(target_skeleton, targets,
                                                         frame_range=frame_range,
                                                         scale_factor=scale_factor,
                                                         use_optimization=use_optimization)

        else:
            new_frames = get_new_frames_using_euler(target_skeleton, targets,
                                                    frame_range=frame_range,
                                                    scale_factor=scale_factor)
    else:
        new_frames = retarget_from_src_to_target(src_skeleton, target_skeleton, src_motion.frames, GAME_ENGINE_TO_ROCKETBOX_MAP, additional_rotation_map,
                                                 scale_factor=scale_factor, frame_range=frame_range)


    return new_frames


def load_target_skeleton(file_path):
    skeleton = None
    if file_path.lower().endswith("bvh"):
        target_bvh = BVHReader(file_path)
        animated_joints = list(target_bvh.get_animated_joints())
        skeleton = SkeletonBuilder().load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)
    return skeleton


def export(skeleton, frames, out_file, use_euler=False, format="bvh"):
    if not use_euler:
        mv = MotionVector()
        mv.frame_time = skeleton.frame_time
        mv.frames = frames
    else:
        mv = create_motion_vector_from_euler_frames(skeleton, frames)

    if format == "bvh":
        mv.export(target_skeleton, ".", out_file, add_time_stamp=False)

def scale_skeleton(skeleton, scale_factor):
    for node in list(skeleton.nodes.values()):
        node.offset = np.array(node.offset) * scale_factor

if __name__ == "__main__":

    src_file = "skeleton.bvh"
    target_file = "game_engine_target.fbx"
    out_file = "out8"
    export_format = "bvh"
    frame_range = None
    use_optimization = False
    use_euler = False
    root_scale_factor = 1# 0.08815605958679036
    skeleton_scale = 8.815605958679036

    src_bvh = BVHReader(src_file)
    src_skeleton = SkeletonBuilder().load_from_bvh(src_bvh, add_tool_joints=False)
    src_motion = MotionVector()
    src_motion.from_bvh_reader(src_bvh)
    target_skeleton = load_target_skeleton(target_file)
    if target_skeleton is not None:

        scale_skeleton(target_skeleton, skeleton_scale)

        new_frames = retarget(src_skeleton, src_motion, target_skeleton,
                              inv_joint_map=ROCKETBOX_TO_GAME_ENGINE_MAP,
                              additional_rotation_map=ADDITIONAL_ROTATION_MAP,
                              frame_range=frame_range,
                              scale_factor=root_scale_factor,
                              use_optimization=use_optimization,
                              use_euler=use_euler)
        export(target_skeleton,new_frames,out_file, use_euler, export_format)
    else:
        print("Error: could not read target skeleton", target_file)
