import collections


# Edinburgh_skeleton = collections.OrderedDict(
#     [
#         ('Hips', {'parent': None, 'index': 0}),
#         ('LeftUpLeg', {'parent': 'Hips', 'index': 1}),
#         ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 2}),
#         ('LeftFoot', {'parent': 'LeftLeg', 'index': 3}),
#         ('LeftToeBase', {'parent': 'LeftFoot', 'index': 4}),
#         ('RightUpLeg', {'parent': 'Hips', 'index': 5}),
#         ('RightLeg', {'parent': 'RightUpLeg', 'index': 6}),
#         ('RightFoot', {'parent': 'RightLeg', 'index': 7}),
#         ('RightToeBase', {'parent': 'RightFoot', 'index': 8}),
#         ('Spine', {'parent': 'Hips', 'index': 9}),
#         ('Spine1', {'parent': 'Spine', 'index': 10}),
#         ('Neck1', {'parent': 'Spine1', 'index': 11}),
#         ('Head', {'parent': 'Neck1', 'index': 12}),
#         ('LeftArm', {'parent': 'Spine1', 'index': 13}),
#         ('LeftForeArm', {'parent': 'LeftArm', 'index': 14}),
#         ('LeftHand', {'parent': 'LeftForeArm', 'index': 15}),
#         ('LeftHandIndex1', {'parent': 'LeftHand', 'index': 16}),
#         ('RightArm', {'parent': 'Spine1', 'index': 17}),
#         ('RightForeArm', {'parent': 'RightArm', 'index': 18}),
#         ('RightHand', {'parent': 'RightForeArm', 'index': 19}),
#         ('RightHandIndex1', {'parent': 'RightHand', 'index': 20})
#     ]
# )
GAME_ENGINE_ANIMATED_JOINTS = ['Game_engine', 'Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l', 'lowerarm_l',
                               'hand_l', 'clavicle_r',
                               'upperarm_r', 'lowerarm_r', 'hand_r', 'neck_01', 'head', 'thigh_l', 'calf_l', 'foot_l',
                               'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']


GAME_ENGINE_ANIMATED_JOINTS_without_game_engine = ['Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l', 'lowerarm_l',
                                                   'hand_l', 'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r',
                                                   'neck_01', 'head', 'thigh_l', 'calf_l', 'foot_l',
                                                   'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']

Edinburgh_ANIMATED_JOINTS = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg',
                             'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck1', 'Head', 'LeftArm', 'LeftForeArm',
                             'LeftHand', 'LeftHandIndex1', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex1']


Edinburgh_skeleton = [
        {'name': 'Hips', 'parent': None, 'index': 0},
        {'name': 'LeftUpLeg', 'parent': 'Hips', 'index': 1},
        {'name': 'LeftLeg', 'parent': 'LeftUpLeg', 'index': 2},
        {'name': 'LeftFoot', 'parent': 'LeftLeg', 'index': 3},
        {'name': 'LeftToeBase', 'parent': 'LeftFoot', 'index': 4},
        {'name': 'RightUpLeg', 'parent': 'Hips', 'index': 5},
        {'name': 'RightLeg', 'parent': 'RightUpLeg', 'index': 6},
        {'name': 'RightFoot', 'parent': 'RightLeg', 'index': 7},
        {'name': 'RightToeBase', 'parent': 'RightFoot', 'index': 8},
        {'name': 'Spine', 'parent': 'Hips', 'index': 9},
        {'name': 'Spine1', 'parent': 'Spine', 'index': 10},
        {'name': 'Neck1', 'parent': 'Spine1', 'index': 11},
        {'name': 'Head', 'parent': 'Neck1', 'index': 12},
        {'name': 'LeftArm', 'parent': 'Spine1', 'index': 13},
        {'name': 'LeftForeArm', 'parent': 'LeftArm', 'index': 14},
        {'name': 'LeftHand', 'parent': 'LeftForeArm', 'index': 15},
        {'name': 'LeftHandIndex1', 'parent': 'LeftHand', 'index': 16},
        {'name': 'RightArm', 'parent': 'Spine1', 'index': 17},
        {'name': 'RightForeArm', 'parent': 'RightArm', 'index': 18},
        {'name': 'RightHand', 'parent': 'RightForeArm', 'index': 19},
        {'name': 'RightHandIndex1', 'parent': 'RightHand', 'index': 20}
    ]

Edinburgh_skeleton_full = [
        {'name': 'Hips', 'parent': None, 'index': 0},
        {'name': 'LHipJoint', 'parent': 'Hips', 'index': 1},
        {'name': 'LeftUpLeg', 'parent': 'Hips', 'index': 2},
        {'name': 'LeftLeg', 'parent': 'LeftUpLeg', 'index': 3},
        {'name': 'LeftFoot', 'parent': 'LeftLeg', 'index': 4},
        {'name': 'LeftToeBase', 'parent': 'LeftFoot', 'index': 5},
        {'name': 'RightUpLeg', 'parent': 'Hips', 'index': 6},
        {'name': 'RightLeg', 'parent': 'RightUpLeg', 'index': 7},
        {'name': 'RightFoot', 'parent': 'RightLeg', 'index': 8},
        {'name': 'RightToeBase', 'parent': 'RightFoot', 'index': 9},
        {'name': 'Spine', 'parent': 'Hips', 'index': 10},
        {'name': 'Spine1', 'parent': 'Spine', 'index': 11},
        {'name': 'Neck1', 'parent': 'Spine1', 'index': 12},
        {'name': 'Head', 'parent': 'Neck1', 'index': 13},
        {'name': 'LeftArm', 'parent': 'Spine1', 'index': 14},
        {'name': 'LeftForeArm', 'parent': 'LeftArm', 'index': 15},
        {'name': 'LeftHand', 'parent': 'LeftForeArm', 'index': 16},
        {'name': 'LeftHandIndex1', 'parent': 'LeftHand', 'index': 17},
        {'name': 'RightArm', 'parent': 'Spine1', 'index': 18},
        {'name': 'RightForeArm', 'parent': 'RightArm', 'index': 19},
        {'name': 'RightHand', 'parent': 'RightForeArm', 'index': 20},
        {'name': 'RightHandIndex1', 'parent': 'RightHand', 'index': 21}
    ]



GAME_ENGINE_JOINTS_DOFS = {
    "calf_l": ['X', 'Z'],
    "calf_r": ['X', 'Z'],
    "thigh_l": ['X', 'Z'],
    "thigh_r": ['X', 'Z'],
    "foot_l": ['X', 'Z'],
    "foot_r": ['X', 'Z'],
    "ball_l": ['X', 'Z'],
    "ball_r": ['X', 'Z'],
    "clavicle_l": ['X', 'Z'],
    "clavicle_r": ['X', 'Z'],
    "upperarm_l": ['X', 'Z'],
    "upperarm_r": ['X', 'Z'],
    "lowerarm_l": ['X', 'Z'],
    "lowerarm_r": ['X', 'Z'],
    "hand_l": ['X', 'Z'],
    "hand_r": ['X', 'Z'],
    "spine_03": ['X', 'Z']
}