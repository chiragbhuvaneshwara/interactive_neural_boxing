import os
from .animation_data import BVHReader, SkeletonBuilder, MotionVector
from .animation_data.skeleton_models import RAW_SKELETON_MODEL
from .animation_data.motion_editing.motion_grounding import MotionGrounding

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"


def run_motion_editing(bvh_file):
    ik_chains = RAW_SKELETON_MODEL["ik_chains"]
    bvh = BVHReader(bvh_file)
    #animated_joints = list(bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh)# filter here
    mv = MotionVector()
    mv.from_bvh_reader(bvh, True) # filter here
    ik_config = {
        "tolerance": 0.05,
        "optimization_method": "L-BFGS-B",
        "max_iterations": 1000,
        "interpolation_window": 120,
        "transition_window": 60,
        "use_euler_representation": False,
        "solving_method": "unconstrained",
        "activate_look_at": True,
        "max_retries": 5,
        "success_threshold": 5.0,
        "optimize_orientation": True,
        "elementary_action_max_iterations": 5,
        "elementary_action_optimization_eps": 1.0,
        "adapt_hands_during_carry_both": True,
        "constrain_place_orientation": False,
        "activate_blending": True,
        "version": 1,
        "use_fabrik": True
    }
    me = MotionGrounding(skeleton, ik_config, ik_chains)
    position = [10, 130, -40]
    #position = [10, 20, -40]
    direction = [1,0,0]
    me.add_constraint("RightHand", [0,100], position, direction)
    mv.frames = me.run(mv)
    print("export motion")
    mv.frames = skeleton.add_fixed_joint_parameters_to_motion(mv.frames)
    mv.export(skeleton, "out" + os.sep + "out")

if __name__ == "__main__":
    bvh_file = "foot_sliding_example.bvh"
    run_motion_editing(bvh_file)
