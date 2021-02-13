import numpy as np


class TPosture(object):
    """
    Attributes:
     - bones
     - bone_map
     - location
     - rotation

    """

    def __init__(self, bones=None, bone_map=None, location=None, rotation=None, arm_tr=None):
        self.bones = bones
        self.bone_map = bone_map
        self.location = location
        self.rotation = rotation
        self.arm_tr = arm_tr


class TBone(object):
    """
    Attributes:
     - name
     - position
     - rotation
     - children
     - parent

    """

    def __init__(self, name=None, position=None, rotation=None, children=None, parent=None, ):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.children = children
        self.parent = parent


class TVector3(object):
    """
    Attributes:
     - x
     - y
     - z

    """

    def __init__(self, x=None, y=None, z=None, ):
        self.x = x
        self.y = y
        self.z = z


class TQuaternion(object):
    """
    Attributes:
     - w
     - x
     - y
     - z

    """

    def __init__(self, w=None, x=None, y=None, z=None, ):
        self.w = w
        self.x = x
        self.y = y
        self.z = z


def TVector3_2np(x):
    return np.array([x.x, x.y, x.z])


def np_2TVector3(x):
    return TVector3(x[0], x[1], x[2])


def build_zero_posture(base_controller, position_str="position"):
    zp = base_controller.zero_posture
    bonelist = []
    mapping = {}
    for bone in zp:
        mapping[bone["name"]] = (bone["index"])
    for bone in zp:
        children = [c for c in bone["children"]]

        position = np.array([float(bone[position_str][0]), float(bone[position_str][1]), float(bone[position_str][2])])
        if "local" in position_str:
            position = position / 100
        else:
            position = position
        position = np_2TVector3(position)

        rotation = TQuaternion(float(bone["local_rotation"][0]), float(bone["local_rotation"][1]),
                               float(bone["local_rotation"][2]), float(bone["local_rotation"][3]))
        tb = TBone(bone["name"], position, rotation, children, bone["parent"])
        bonelist.append(tb)

    arm_tr = {'rwt': [0]*10, 'lwt': [0]*10}

    return TPosture(bonelist, mapping, [0,0,0], 0.0, arm_tr)


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    # if isinstance(obj, TBone):
        # print(obj.__dict__)

    return obj.__dict__
