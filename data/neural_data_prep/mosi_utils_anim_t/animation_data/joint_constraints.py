import numpy as np
from math import acos, cos, sin, atan
from ..external.transformations import quaternion_about_axis, quaternion_inverse, quaternion_multiply

def exponential(q):
    """ https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation
        chapter 2 of lee 2000
    """
    v = np.array([q[1], q[2], q[3]])
    theta = np.linalg.norm(v)
    return (cos(theta) + v/ theta * sin(theta))# * np.exp(q[0])

def log(q):
    """ https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation
      chapter 2 of lee 2000
    """
    v = np.array([q[1], q[2], q[3]])
    w = q[0]
    if w == 0:
        return np.pi/2 * v
    elif 0 < abs(w) < 1:
        theta = np.linalg.norm(v)
        return v / theta * atan(theta/w)
    else: # w == 1
        return 0

def dist(q1, q2):
    """  chapter 2 of lee 2000"""
    return np.linalg.norm(log(quaternion_multiply(quaternion_inverse(q1), q2)))

def rotate_vector_by_quaternion(v, q):
    vq = [0, v[0], v[1], v[2]]
    v_prime = quaternion_multiply(q, quaternion_multiply(vq , quaternion_inverse(q)))[1:]
    v_prime /= np.linalg.norm(v_prime)
    return v_prime


def apply_conic_constraint(q, ref_q, axis, k):
    """ lee 2000 p. 48"""
    q0 = ref_q
    w = axis
    #rel_q = quaternion_multiply(quaternion_inverse(q0), q)
    #rel_q /np.linalg.norm(rel_q)
    w_prime = rotate_vector_by_quaternion(w, q)
    w_prime = normalize(w_prime)
    phi = acos(np.dot(w, w_prime))
    if 0 < phi > k: # apply
        delta_phi = k - phi
        #print(delta_phi)
        v = np.cross(w, w_prime)
        delta_q = quaternion_about_axis(delta_phi, v)
        q = quaternion_multiply(delta_q, q)
    return q


def apply_axial_constraint(q, ref_q, axis, k1, k2):
    """ lee 2000 p. 48"""
    q0 = ref_q
    w = axis
    rel_q = quaternion_multiply(quaternion_inverse(q0), q)
    rel_q /np.linalg.norm(rel_q)
    w_prime = rotate_vector_by_quaternion(w, rel_q)
    v = np.cross(w, w_prime)
    phi = acos(np.dot(w, w_prime))

    #remove rotation around v
    # psi_q = exp(-phi*v)
    inv_phi_q = quaternion_about_axis(-phi, v)
    psi_q = quaternion_multiply(inv_phi_q, ref_q)

    # get angle around w from psi_q
    w_prime2 = rotate_vector_by_quaternion(w, psi_q)
    w_prime2 /= np.linalg.norm(w_prime2)
    psi = acos(np.dot(w, w_prime2))

    if k1 < psi > k2: # apply
        delta_psi1 = k1 - psi
        delta_psi2 = k2 - psi
        delta_psi = min(delta_psi1, delta_psi2)
        delta_q = quaternion_about_axis(delta_psi, w)
        q = quaternion_multiply(delta_q, q)
    return q


def apply_spherical_constraint(q, ref_q, axis, k):
    """ lee 2000 p. 48"""
    q0 = ref_q
    w = axis
    rel_q = quaternion_multiply(quaternion_inverse(q0), q)
    rel_q /np.linalg.norm(rel_q)
    w_prime = rotate_vector_by_quaternion(w, rel_q)
    v = np.cross(w, w_prime)
    log_rel_q = log(rel_q)
    phi = acos(np.dot(w, w_prime))
    if 2*np.linalg.norm(log_rel_q) > k: # apply
        delta_phi = k - phi
        delta_q = quaternion_about_axis(delta_phi, v)
        q = quaternion_multiply(delta_q, q)
    return q

REFERENCE_QUATERNION = [1,0,0,0]


class HingeConstraint(object):
    def __init__(self, axis, deg_angle_range):
        self.axis = axis
        self.angle_range = np.radians(deg_angle_range)

    def apply(self, q, parent_q):
        #axis = rotate_vector_by_quaternion(self.axis, parent_q)
        return apply_axial_constraint(q, parent_q, self.axis, self.angle_range[0], self.angle_range[1])


import math

def normalize(v):
    return v/np.linalg.norm(v)

def quaternion_to_axis_angle(q):
    """http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/

    """
    a = 2* math.acos(q[0])
    s = math.sqrt(1- q[0]*q[0])
    if s < 0.001:
        x = q[1]
        y = q[2]
        z = q[3]
    else:
        x = q[1] / s
        y = q[2] / s
        z = q[3] / s
    v = np.array([x,y,z])
    if np.sum(v)> 0:
        return normalize(v),a
    else:
        return v, a



def swing_twist_decomposition(q, twist_axis):
    """ code by janis sprenger based on
        Dobrowsolski 2015 Swing-twist decomposition in Clifford algebra. https://arxiv.org/abs/1506.05481
    """
    q = normalize(q)
    #twist_axis = np.array((q * offset))[0]
    projection = np.dot(twist_axis, np.array([q[1], q[2], q[3]])) * twist_axis
    twist_q = np.array([q[0], projection[0], projection[1],projection[2]])
    if np.linalg.norm(twist_q) == 0:
        twist_q = np.array([1,0,0,0])
    twist_q = normalize(twist_q)
    swing_q = quaternion_multiply(q, quaternion_inverse(twist_q))#q * quaternion_inverse(twist)
    return swing_q, twist_q

OPENGL_UP = np.array([0,1,0])


class JointConstraint(object):
    def __init__(self):
        self.is_static = False

    def get_axis(self):
        return OPENGL_UP

    def apply(self, q):
        return q

class BallSocketConstraint(JointConstraint):
    def __init__(self, axis, k):
        JointConstraint.__init__(self)
        self.axis = axis
        self.k = k

    def apply(self, q):
        ref_q = [1,0,0,0]
        return apply_spherical_constraint(q, ref_q, self.axis, self.k)


    def get_axis(self):
        return self.axis


class ConeConstraint(JointConstraint):
    def __init__(self, axis, k):
        JointConstraint.__init__(self)
        self.axis = axis
        self.k = k

    def apply(self, q):
        ref_q = [1, 0, 0, 0]
        return apply_conic_constraint(q, ref_q, self.axis, self.k)

    def get_axis(self):
        return self.axis

class HingeConstraint2(JointConstraint):
    def __init__(self, swing_axis, twist_axis, deg_angle_range=None, verbose=False):
        JointConstraint.__init__(self)
        self.swing_axis = swing_axis
        self.twist_axis = twist_axis
        if deg_angle_range is not None:
            self.angle_range = np.radians(deg_angle_range)
        else:
            self.angle_range = None
        self.verbose = False

    def apply(self, q):
        sq, tq = swing_twist_decomposition(q, self.swing_axis)
        tv, ta = quaternion_to_axis_angle(tq)
        if self.verbose:
            print("before", np.degrees(ta), tv)
        if self.angle_range is not None:
            ta = max(ta, self.angle_range[0])
            ta = min(ta, self.angle_range[1])

        if self.verbose:
            print("after", np.degrees(ta), tv)
        return quaternion_about_axis(ta, self.swing_axis)

    def apply_global(self, pm, q):
        axis = np.dot(pm, self.axis)
        axis = normalize(axis)
        sq, tq = swing_twist_decomposition(q, axis)
        tv, ta = quaternion_to_axis_angle(tq)
        if self.verbose:
            print("before", np.degrees(ta), tv)
        if self.angle_range is not None:
            ta = max(ta, self.angle_range[0])
            ta = min(ta, self.angle_range[1])

        if self.verbose:
            print("after", np.degrees(ta), tv)
        return quaternion_about_axis(ta, self.swing_axis)

    def split(self,q):
        axis = normalize(self.swing_axis)
        sq, tq = swing_twist_decomposition(q, axis)
        return sq, tq

    def split_correct(self,q):
        axis = normalize(self.twist_axis)
        sq, tq = swing_twist_decomposition(q, axis)
        return sq, tq

    def get_axis(self):
        return self.swing_axis


class ShoulderConstraint(JointConstraint):
    """ combines conic and axial"""
    def __init__(self, axis, k1,k2, k):
        JointConstraint.__init__(self)
        self.axis = axis
        self.k1 = k1
        self.k2 = k2
        self.k = k

    def apply(self, q):
        ref_q = [1, 0, 0, 0]
        q = apply_conic_constraint(q, ref_q, self.axis, self.k)
        q = normalize(q)
        #q = apply_axial_constraint(q, ref_q, self.axis, self.k1, self.k2)
        return q

    def get_axis(self):
        return self.axis




