import numpy as np


class TPosture(object):

    def __init__(self, bones=None, bone_map=None, location=None, rotation=None, trajectory=None):
        """

        :param bones:
        :param bone_map:
        :param location:
        :param rotation:
        :param trajectory:
        """
        self.bones = bones
        self.bone_map = bone_map
        self.location = location
        self.rotation = rotation
        self.traj = trajectory


class TBone(object):

    def __init__(self, name=None, position=None, rotation=None, children=None, parent=None, ):
        """

        :param name:
        :param position:
        :param rotation:
        :param children:
        :param parent:
        """
        self.name = name
        self.position = position
        self.rotation = rotation
        self.children = children
        self.parent = parent


class TVector3(object):

    def __init__(self, x=None, y=None, z=None, ):
        """

        :param x:
        :param y:
        :param z:
        """
        self.x = x
        self.y = y
        self.z = z


class TQuaternion(object):

    def __init__(self, w=None, x=None, y=None, z=None, ):
        """

        :param w:
        :param x:
        :param y:
        :param z:
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z


def tvector3_to_np(x):
    x = [x[0], x[1], -x[2]]
    return x


def np_to_tvector3(x, vis=False):
    if not vis:
        return TVector3(x[0], x[1], -x[2])
    else:
        return TVector3(x[0], x[1], x[2])


def build_zero_posture(base_controller, position_str="position", num_traj_pts = 10):
    """

    :param base_controller:
    :param position_str:
    :return:
    """
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
        position = np_to_tvector3(position)

        rotation = TQuaternion(float(bone["local_rotation"][0]), float(bone["local_rotation"][1]),
                               float(bone["local_rotation"][2]), float(bone["local_rotation"][3]))
        tb = TBone(bone["name"], position, rotation, children, bone["parent"])
        bonelist.append(tb)

    ntp = num_traj_pts
    traj = {'rt': [0] * ntp,
            'rt_v': [0] * ntp,
            'rwt': [0] * ntp,
            'lwt': [0] * ntp,
            'rwt_v': [0] * ntp,
            'lwt_v': [0] * ntp}

    return TPosture(bonelist, mapping, [0, 0, 0], 0.0, traj)


def serialize(obj):
    """
    JSON serializer for objects not serializable by default json code
    :param obj:
    :return:
    """
    return obj.__dict__
