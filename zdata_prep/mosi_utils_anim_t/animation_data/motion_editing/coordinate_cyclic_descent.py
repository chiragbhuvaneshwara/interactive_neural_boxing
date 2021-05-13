import numpy as np
from ...external.transformations import quaternion_from_matrix, quaternion_matrix
from ..joint_constraints import quaternion_to_axis_angle


def normalize(v):
    return v / np.linalg.norm(v)


def to_local_coordinate_system(skeleton, frame, joint_name, q):
    """ given a global rotation bring it to the local coordinate system"""
    if skeleton.nodes[joint_name].parent is not None:
        global_m = quaternion_matrix(q)
        parent_joint = skeleton.nodes[joint_name].parent.node_name
        parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
        new_local = np.dot(np.linalg.inv(parent_m), global_m)
        return quaternion_from_matrix(new_local)
    else:
        return q


def quaternion_from_vector_to_vector(a, b):
    """src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    http://wiki.ogre3d.org/Quaternion+and+Rotation+Primer"""

    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    if np.dot(q,q) != 0:
        return q/ np.linalg.norm(q)
    else:
        idx = np.nonzero(a)[0]
        q = np.array([0, 0, 0, 0])
        q[1 + ((idx + 1) % 2)] = 1 # [0, 0, 1, 0] for a rotation of 180 around y axis
        return q


def orient_end_effector_to_target(skeleton, root, end_effector, frame, constraint):
    """ find angle between the vectors end_effector - root and target- root """

    # align vectors
    root_pos = skeleton.nodes[root].get_global_position(frame)
    if constraint.offset is not None:
        m = skeleton.nodes[end_effector].get_global_matrix(frame)
        end_effector_pos = np.dot(m, constraint.offset)[:3]
    else:
        end_effector_pos = skeleton.nodes[end_effector].get_global_position(frame)

    src_delta = end_effector_pos - root_pos
    src_dir = src_delta / np.linalg.norm(src_delta)

    target_delta = constraint.position - root_pos
    target_dir = target_delta / np.linalg.norm(target_delta)

    root_delta_q = quaternion_from_vector_to_vector(src_dir, target_dir)
    root_delta_q = normalize(root_delta_q)

    global_m = quaternion_matrix(root_delta_q)
    parent_joint = skeleton.nodes[root].parent.node_name
    parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
    old_global = np.dot(parent_m, skeleton.nodes[root].get_local_matrix(frame))
    new_global = np.dot(global_m, old_global)
    q = quaternion_from_matrix(new_global)
    return normalize(q)


def orient_node_to_target(skeleton,frame,node_name, end_effector, constraint):
    #o = skeleton.animated_joints.index(node_name) * 4 + 3
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = orient_end_effector_to_target(skeleton, node_name, end_effector, frame, constraint)
    q = to_local_coordinate_system(skeleton, frame, node_name, q)
    frame[o:o + 4] = q
    return frame


def apply_joint_constraint(skeleton,frame,node_name):
    #o = skeleton.animated_joints.index(node_name) * 4 + 3
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = np.array(frame[o:o + 4])
    q = skeleton.nodes[node_name].joint_constraint.apply(q)
    frame[o:o + 4] = q
    return frame

def set_global_orientation(skeleton, frame, node_name, orientation):
    m = quaternion_matrix(orientation)
    #o = skeleton.animated_joints.index(node_name) * 4 + 3
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    parent_m = skeleton.nodes[node_name].parent.get_global_matrix(frame, use_cache=False)
    local_m = np.dot(np.linalg.inv(parent_m), m)
    q = quaternion_from_matrix(local_m)
    frame[o:o + 4] = normalize(q)
    return frame


def run_ccd(skeleton, frame, end_effector_name, constraint, eps=0.01, n_max_iter=50, chain_end_joint=None, verbose=False):
    pos = skeleton.nodes[end_effector_name].get_global_position(frame)
    error = np.linalg.norm(constraint.position-pos)
    n_iters = 0
    while error > eps and n_iters < n_max_iter:
        node = skeleton.nodes[end_effector_name].parent
        depth = 0

        while node is not None and node.node_name != chain_end_joint and node.parent is not None:
            static = False
            if node.joint_constraint is not None and node.joint_constraint.is_static:
                static = True

            if not static:
                frame = orient_node_to_target(skeleton,frame, node.node_name, end_effector_name, constraint)
                if constraint.orientation is not None:
                    frame = set_global_orientation(skeleton, frame, end_effector_name, constraint.orientation)

                if node.joint_constraint is not None:
                    frame = apply_joint_constraint(skeleton, frame, node.node_name)
            node = node.parent
            depth += 1

        if constraint.offset is not None:
            m = skeleton.nodes[end_effector_name].get_global_matrix(frame)
            end_effector_pos = np.dot(m, constraint.offset)[:3]
        else:
            end_effector_pos = skeleton.nodes[end_effector_name].get_global_position(frame)

        error = np.linalg.norm(constraint.position - end_effector_pos)
        n_iters += 1

    if verbose:
            print("error at", n_iters, ":", error, "c:",constraint.position,"pos:", pos)
    return frame, error


LOOK_AT_DIR = [0, -1,0]
SPINE_LOOK_AT_DIR = [0,0,1]

def look_at_target(skeleton, root, end_effector, frame, position, local_dir=LOOK_AT_DIR):
    """ find angle between the look direction and direction between end effector and target"""
    #direction of endeffector
    m = skeleton.nodes[end_effector].get_global_matrix(frame)
    #offset = skeleton.nodes[end_effector].offset
    end_effector_dir = np.dot(m[:3,:3], local_dir)
    end_effector_dir = end_effector_dir / np.linalg.norm(end_effector_dir)

    # direction from endeffector to target
    end_effector_pos = m[:3, 3]
    target_delta = position - end_effector_pos
    target_dir = target_delta / np.linalg.norm(target_delta)

    # find rotation to align vectors
    root_delta_q = quaternion_from_vector_to_vector(end_effector_dir, target_dir)
    root_delta_q = normalize(root_delta_q)

    #apply global delta to get new global matrix of joint
    global_m = quaternion_matrix(root_delta_q)
    parent_joint = skeleton.nodes[root].parent.node_name
    parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
    old_global = np.dot(parent_m, skeleton.nodes[root].get_local_matrix(frame))
    new_global = np.dot(global_m, old_global)
    q = quaternion_from_matrix(new_global)
    return normalize(q)



def orient_node_to_target_look_at(skeleton,frame,node_name, end_effector, position,  local_dir=LOOK_AT_DIR):
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = look_at_target(skeleton, node_name, end_effector, frame, position, local_dir)
    q = to_local_coordinate_system(skeleton, frame, node_name, q)
    frame[o:o + 4] = q
    return frame



def run_ccd_look_at(skeleton, frame, end_effector_name, position, eps=0.01, n_max_iter=1, local_dir=LOOK_AT_DIR):
    error = np.inf
    n_iter = 0
    while error > eps and n_iter < n_max_iter:
        node = skeleton.nodes[end_effector_name].parent
        depth = 0
        while node is not None and node.node_name != skeleton.root:
            frame = orient_node_to_target_look_at(skeleton,frame, node.node_name, end_effector_name, position, local_dir)
            if node.joint_constraint is not None:
                frame = apply_joint_constraint(skeleton, frame, node.node_name)
            node = node.parent
            depth += 1

        m = skeleton.nodes[end_effector_name].get_global_matrix(frame)

        end_effector_dir = np.dot(m[:3, :3], local_dir)
        end_effector_dir = end_effector_dir / np.linalg.norm(end_effector_dir)

        # direction from endeffector to target
        end_effector_pos = m[:3,3]
        target_delta = position - end_effector_pos
        target_dir = target_delta / np.linalg.norm(target_delta)
        root_delta_q = quaternion_from_vector_to_vector(end_effector_dir, target_dir)
        root_delta_q = normalize(root_delta_q)
        v, a = quaternion_to_axis_angle(root_delta_q)
        error = abs(a)
        #error = np.linalg.norm(target_dir-end_effector_dir)
        print(error)
        n_iter+=1

    #if verbose:
    print("error at", n_iter, ":", error, "c:",position,"pos:")

    return frame, error


