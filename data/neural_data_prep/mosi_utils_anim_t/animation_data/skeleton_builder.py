import collections
from copy import deepcopy
import json
import numpy as np
from .skeleton import Skeleton
from .skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE
from .skeleton_models import ROCKETBOX_ANIMATED_JOINT_LIST, ROCKETBOX_FREE_JOINTS_MAP, ROCKETBOX_REDUCED_FREE_JOINTS_MAP, ROCKETBOX_SKELETON_MODEL, ROCKETBOX_BOUNDS, ROCKETBOX_TOOL_BONES, ROCKETBOX_ROOT_DIR
from .quaternion_frame import QuaternionFrame


def create_identity_frame(skeleton):
    skeleton.identity_frame = np.zeros(skeleton.reference_frame_length)
    offset = 3
    for j in list(skeleton.nodes.keys()):
        if skeleton.nodes[j].index > 0:
            skeleton.identity_frame[offset:offset + 4] = [1, 0, 0, 0]
            offset += 4


def create_euler_frame_indices(skeleton):
    nodes_without_endsite = [node for node in list(skeleton.nodes.values()) if node.node_type != SKELETON_NODE_TYPE_END_SITE]
    for node in nodes_without_endsite:
        node.euler_frame_index = nodes_without_endsite.index(node)


def read_reference_frame_from_bvh_reader(bvh_reader, frame_index=0):
    quat_frame = QuaternionFrame(bvh_reader, bvh_reader.frames[frame_index], False, False, animated_joints=None)
    quat_frames = list(quat_frame.values())
    quaternion_frame = np.array(quat_frames).flatten()
    return np.array(bvh_reader.frames[0][:3].tolist() + quaternion_frame.tolist())


def add_tool_nodes(skeleton, node_names, new_tool_bones):
    for b in new_tool_bones:
        add_new_end_site(skeleton, node_names, b["parent_node_name"], b["new_node_offset"])
        skeleton.tool_nodes.append(b["new_node_name"])


def add_new_end_site(skeleton, node_names, parent_node_name, offset):
    if parent_node_name in list(node_names.keys()):
        level = node_names[parent_node_name]["level"] + 1
        node_desc = dict()
        node_desc["level"] = level
        node_desc["offset"] = offset


def generate_reference_frame(skeleton, animated_joints):
    identity_frame = [0,0,0]
    frame = [0, 0, 0]
    joint_idx = 0
    node_list = [(skeleton.nodes[n].index, n) for n in skeleton.nodes.keys() if skeleton.nodes[n].index >= 0]
    node_list.sort()
    for idx, node in node_list:
        frame += list(skeleton.nodes[node].rotation)
        if node in animated_joints:
            identity_frame += [1.0,0.0,0.0,0.0]
            skeleton.nodes[node].quaternion_frame_index = joint_idx
            joint_idx += 1
        else:
            skeleton.nodes[node].quaternion_frame_index = -1
    skeleton.reference_frame = np.array(frame)
    skeleton.reference_frame_length = len(frame)
    skeleton.identity_frame = np.array(identity_frame)


class SkeletonBuilder(object):

    def load_from_bvh(self, bvh_reader, animated_joints=None, add_tool_joints=True, reference_frame=None, skeleton_model=None):
        skeleton = Skeleton()
        if animated_joints is None:
            animated_joints = [joint for joint in bvh_reader.get_animated_joints()]
        skeleton.animated_joints = animated_joints
        skeleton.frame_time = deepcopy(bvh_reader.frame_time)
        skeleton.root = deepcopy(bvh_reader.root)
        skeleton.aligning_root_node = skeleton.root
        skeleton.aligning_root_dir = ROCKETBOX_ROOT_DIR
        if reference_frame is None:
            skeleton.reference_frame = read_reference_frame_from_bvh_reader(bvh_reader)
        else:
            skeleton.reference_frame = reference_frame
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        skeleton.node_channels = collections.OrderedDict()
        skeleton.nodes = collections.OrderedDict()
        skeleton.tool_nodes = []
        if add_tool_joints:
            add_tool_nodes(skeleton, bvh_reader.node_names, ROCKETBOX_TOOL_BONES)
        skeleton.max_level = skeleton._get_max_level()
        skeleton._set_joint_weights()
        skeleton.nodes = collections.OrderedDict()
        joint_list = [k for k in bvh_reader.node_names if
                      "children" in list(bvh_reader.node_names[k].keys()) and
                      len(bvh_reader.node_names[k]["children"]) > 0]
        self.construct_hierarchy_from_bvh(skeleton, joint_list, bvh_reader.node_names, skeleton.root, 0)

        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()
        create_euler_frame_indices(skeleton)
        create_identity_frame(skeleton)
        if skeleton_model is not None:
            skeleton.skeleton_model = skeleton_model
            skeleton.add_heels(skeleton_model)
        return skeleton

    def construct_hierarchy_from_bvh(self, skeleton, joint_list, node_info, node_name, level):
        if "channels" in node_info[node_name]:
            channels = node_info[node_name]["channels"]
        else:
            channels = []
        if "channel_indices" in node_info[node_name]:
            channel_indices = node_info[node_name]["channel_indices"]
        else:
            channel_indices = []
        if node_name in joint_list:
            joint_index = joint_list.index(node_name)
        else:
            joint_index = -1
        if node_name == skeleton.root:
            node = SkeletonRootNode(node_name, channels, None, level, channel_indices)
            if node_name in skeleton.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            else:
                node.fixed = True
            node.index = joint_index
        elif "children" in list(node_info[node_name].keys()) and len(node_info[node_name]["children"]) > 0:
            node = SkeletonJointNode(node_name, channels, None, level, channel_indices)
            if node_name in skeleton.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            else:
                node.fixed = True

            offset = joint_index * 4 + 3
            node.rotation = skeleton.reference_frame[offset: offset + 4]
            node.index = joint_index
        else:
            node = SkeletonEndSiteNode(node_name, channels, None, level)

        node.index = joint_index
        node.offset = node_info[node_name]["offset"]
        skeleton.nodes[node_name] = node
        if "children" in list(node_info[node_name].keys()):
            for c in node_info[node_name]["children"]:
                c_node = self.construct_hierarchy_from_bvh(skeleton, joint_list, node_info, c, level+1)
                c_node.parent = node
                node.children.append(c_node)
        return node

    def load_from_json_file(self, filename):
        with open(filename) as infile:
            data = json.load(infile)
            skeleton = self.load_from_json_data(data)
            return skeleton

    def load_from_json_data(self, data, animated_joints=None):
        def extract_animated_joints(node, animated_joints):
            animated_joints.append(node["name"])
            for c in node["children"]:
                if c["index"] >= 0:
                    extract_animated_joints(c, animated_joints)

        skeleton = Skeleton()
        print("load from json")
        if animated_joints is not None:
            skeleton.animated_joints = animated_joints
        elif "animated_joints" in data.keys():
            skeleton.animated_joints = data["animated_joints"]
        else:
            animated_joints = list()
            extract_animated_joints(data["root"], animated_joints)
        if "free_joints_map" in list(data.keys()):
            skeleton.free_joints_map = data["free_joints_map"]
        skeleton.reduced_free_joints_map = ROCKETBOX_REDUCED_FREE_JOINTS_MAP  # data["reduced_free_joints_map"]
        skeleton.bounds = ROCKETBOX_BOUNDS  # data["bounds"]
        if "skeleton_model" in list(data.keys()):
            skeleton.skeleton_model = data["skeleton_model"]
        else:
            skeleton.skeleton_model = collections.OrderedDict()
            skeleton.skeleton_model["joints"] = dict()
            if "head_joint" in list(data.keys()):
                skeleton.skeleton_model["joints"]["head"] = data["head_joint"]
            if "neck_joint" in list(data.keys()):
                skeleton.skeleton_model["joints"]["neck"] = data["neck_joint"]

        skeleton.frame_time = data["frame_time"]
        skeleton.nodes = collections.OrderedDict()
        root = self._create_node_from_desc(skeleton, data["root"], None, 0)
        skeleton.root = root.node_name
        if "tool_nodes" in list(data.keys()):
            skeleton.tool_nodes = data["tool_nodes"]
        skeleton.max_level = skeleton._get_max_level()
        skeleton._set_joint_weights()
        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()
        create_euler_frame_indices(skeleton)
        if "aligning_root_node" in list(data.keys()):
            skeleton.aligning_root_node = data["aligning_root_node"]
        else:
            skeleton.aligning_root_node = skeleton.root
        if "aligning_root_dir" in list(data.keys()):
            skeleton.aligning_root_dir = data["aligning_root_dir"]
        else:
            skeleton.aligning_root_dir = ROCKETBOX_ROOT_DIR

        generate_reference_frame(skeleton, skeleton.animated_joints)
        return skeleton

    def load_from_fbx_data(self, data):
        skeleton = Skeleton()
        skeleton.nodes = collections.OrderedDict()

        skeleton.animated_joints = data["animated_joints"]
        # self.inv_bind_poses = [self._create_node_from_desc(node, None) for node in data["nodes"].values()]
        skeleton.root = data["root"]
        self._create_node_from_desc2(skeleton, data, skeleton.root, None, 0)
        skeleton.frame_time = data["frame_time"]
        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()

        n_params = len(skeleton.animated_joints) * 4 + 3
        skeleton.reference_frame = np.zeros(n_params)
        offset = 3
        for node_name in skeleton.animated_joints:
            skeleton.reference_frame[offset:offset + 4] = data["nodes"][node_name]["rotation"]
            offset += 4
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        create_identity_frame(skeleton)
        return skeleton

    def _create_node_from_desc(self, skeleton, data, parent, level):
        node_name = data["name"]
        channels = data["channels"]
        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent, level)
        elif data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent, level)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent, level)
        # node.fixed = data["fixed"]
        node.index = data["index"]
        node.offset = np.array(data["offset"])
        node.rotation = np.array(data["rotation"])
        if node_name in skeleton.animated_joints:
            node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            node.fixed = False
        else:
            node.quaternion_frame_index = -1
            node.fixed = True

        skeleton.nodes[node_name] = node
        skeleton.nodes[node_name].children = []
        for c_desc in data["children"]:
            skeleton.nodes[node_name].children.append(self._create_node_from_desc(skeleton, c_desc, node, level+1))
        return node

    def _create_node_from_desc2(self, skeleton, data, node_name, parent, level=0):
        node_data = data["nodes"][node_name]
        channels = node_data["channels"]
        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent, level)
        elif node_data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent, level)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent, level)
        node.fixed = node_data["fixed"]
        node.index = node_data["index"]
        node.offset = np.array(node_data["offset"])
        node.rotation = np.array(node_data["rotation"])
        if node_name in skeleton.animated_joints:
            node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
        else:
            node.quaternion_frame_index = -1
        node.children = []
        skeleton.nodes[node_name] = node
        for c_name in node_data["children"]:
            c_node = self._create_node_from_desc2(skeleton, data, c_name, node, level+1)
            node.children.append(c_node)
        return node

