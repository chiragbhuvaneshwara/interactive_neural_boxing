# encoding: UTF-8
import numpy as np
from ..external.transformations import *


class Quaternion(object):
    """
    a wrapper for functions in transformation.py
    Quaternions w+ix+jy+kz are represented as [w, x, y, z]
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self.w = 1; self.x = 0; self.y = 0; self.z = 0
        elif len(args) == 1 and isinstance(args[0], Quaternion):
            self.w = args[0].w
            self.x = args[0].x
            self.y = args[0].y
            self.z = args[0].z
        elif len(args) == 1 and len(args[0]) == 4:
            self.w = args[0][0]
            self.x = args[0][1]
            self.y = args[0][2]
            self.z = args[0][3]
        elif len(args) == 4:
            self.w = args[0]
            self.x = args[1]
            self.y = args[2]
            self.z = args[3]
        else:
            raise ValueError('Unknown initialization')

    def __str__(self): return str([self.w, self.x, self.y, self.z])

    def __repr__(self): return str([self.w, self.x, self.y, self.z])

    def __add__(self, other): return self * other
    def __sub__(self, other): return self / other

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = -self.x * other.x - other.y * self.y - other.z * self.z + other.w * self.w
            x = self.x * other.w + self.y * other.z - self.z * other.y + self.w * other.x
            y = -self.x * other.z + self.y * other.w + self.z * other.x + self.w * other.y
            z = self.x * other.y - self.y * other.x + self.z * other.w + self.w * other.z
            return Quaternion(w, x, y, z)

        elif isinstance(other, float):
            return Quaternion.slerp(Quaternion.identity(), self, other)

        elif isinstance(other, np.ndarray) and other.shape[-1] == 3:
            v = Quaternion(np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1))
            return (self * (v * -self)).imag
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, Quaternion): return self * (-other)
        elif isinstance(other, float): self * (1.0/other)
        else:
            raise TypeError('Cannot divide/subtract Quaternions with type %s' + str(type(other)))

    def __neg__(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def invert_rotation(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    @property
    def imag(self):
        return np.array([self.x, self.y, self.z])

    @property
    def real(self):
        return np.array(self.w)

    @staticmethod
    def identity():
        return Quaternion()

    def length(self):
        return (self.w**2 + self.x**2 + self.y ** 2 + self.z ** 2)**0.5

    def normalized(self):
        return Quaternion(np.array([self.w, self.x, self.y, self.z])/self.length())

    def toVector(self):
        return np.array([self.w, self.x, self.y, self.z])

    @staticmethod
    def slerp(quat0, quat1, fraction):
        """
        quat0 = Quaternion.slerp(quat0, quat1, 0)
        quat1 = Quaternion.slerp(quat0, quat1, 1)
        :param quat0: Quaternion
        :param quat1: Quaternion
        :param fraction: float
        :return: Quaternion
        """
        quat0 = quat0.normalized()
        quat1 = quat1.normalized()
        if fraction == 0.0:
            return quat0
        elif fraction == 1.0:
            return quat1
        else:
            vec0 = quat0.toVector()
            vec1 = quat1.toVector()
            d = np.dot(vec0, vec1)
            if abs(abs(d) - 1.0) < np.finfo(float).eps:
                return quat0
            if d < 0:
                d = - d
                np.negative(vec1, vec1)
            angle = np.arccos(d)
            if abs(angle) < np.finfo(float).eps:
                return quat0
            isin = 1.0 / np.sin(angle)
    
            vec0 *= np.sin((1.0 - fraction) * angle) * isin
            vec1 *= np.sin(fraction * angle) *isin
            vec0 += vec1
            return Quaternion(vec0).normalized()

    def dot(self, q): return np.dot(self.toVector(), q.toVector())

    def toMat4(self):
        return quaternion_matrix(self.toVector())

    @staticmethod
    def fromAngleAxis(angle, axis):
        # print("Angle must be radian")
        return Quaternion(quaternion_about_axis(angle, axis))

    def toAngleAxis(self):
        q = self.normalized()
        s = np.sqrt(1 - q.real**2)
        if s == 0:
            s = 0.001
        angle = 2.0 * np.arccos(q.real)
        axis = q.imag/s
        return angle, axis

    @staticmethod
    def fromEulerAngles(angles, axes='rxyz'):
        '''
        angles must be in radians
        :param angles:
        :param axes:
        :return:
        '''
        assert(len(angles) == 3)
        return Quaternion(quaternion_from_euler(angles[0], angles[1], angles[2], axes=axes))

    def toEulerAnglesRadian(self, axes='rxyz'):
        #print("Output Euler angles are in radians")
        return euler_from_quaternion(self.toVector(), axes)

    def toEulerAnglesDegree(self, axes='rxyz'):
        return np.rad2deg(euler_from_quaternion(self.toVector(), axes))

    @staticmethod
    def fromMat(matrix):
        return Quaternion(quaternion_from_matrix(matrix))

    def toExpmap(self):
        """
        https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
        :return:
        """
        q = self.normalized()
        theta = 2 * np.arccos(q.w)
        if theta < 1e-3:
            auxillary = 1.0/(0.5 + theta ** 2 / 48)
        else:
            auxillary = theta / np.sin(0.5*theta)
        vx = q.x * auxillary
        vy = q.y * auxillary
        vz = q.z * auxillary
        return np.array([vx, vy, vz])

    @staticmethod
    def fromExpmap(expmap_vec):
        """
        https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
        :param expmap_vec:
        :return:
        """
        theta = np.linalg.norm(expmap_vec)
        if theta < 1e-3:
            auxillary = 0.5 + theta ** 2 / 48
        else:
            auxillary = np.sin(0.5*theta) / theta
        qw = np.cos(theta/2)
        qx = expmap_vec[0]*auxillary
        qy = expmap_vec[1]*auxillary
        qz = expmap_vec[2]*auxillary
        return Quaternion([qw, qx, qy, qz])

    def toLogmap(self):
        q = self.normalized()
        theta = 2 * np.arccos(q.w)
        if theta < 1e-3:
            auxillary = 1.0/(0.5 + theta ** 2 / 48)
        else:
            auxillary = theta / np.sin(0.5*theta)
        vx = q.x * auxillary
        vy = q.y * auxillary
        vz = q.z * auxillary
        return np.array([vx, vy, vz])

    @staticmethod
    def fromLogmap(logmap_vec):
        theta = np.linalg.norm(logmap_vec)
        if theta < 1e-3:
            auxillary = 0.5 + theta ** 2 / 48
        else:
            auxillary = np.sin(0.5*theta) / theta
        qw = np.cos(theta/2)
        qx = logmap_vec[0]*auxillary
        qy = logmap_vec[1]*auxillary
        qz = logmap_vec[2]*auxillary
        return Quaternion([qw, qx, qy, qz])

    @staticmethod
    def between(vec1, vec2):
        '''
        compute rotation between two 3D vectors
        :param vec1: 1d array
        :param vec2: 1d array
        :return:
        '''
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        axis = np.cross(vec1, vec2)
        if np.isclose(np.linalg.norm(axis), 0.0):
            if np.allclose(vec1, vec2):
                angle = 0
                return Quaternion.fromAngleAxis(angle, axis)
            elif np.allclose(vec1, -vec2):
                angle = np.deg2rad(180)
                axis = np.array([0, 1, 0])
                return Quaternion.fromAngleAxis(angle, axis)
            else:
                print("exception")
                print(vec1)
                print(vec2)
                raise NotImplementedError
        else:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return Quaternion.fromAngleAxis(angle, axis)

    @staticmethod
    def get_angle_from_quaternion(q, forward, plane='xz'):
        '''
        the angle is the rotation angle that applying quaternion q on the forward vector, all vectors are projected to
        a reference plane
        :param q: Quaternion
        :param forward: reference vector
        :param plane: projection plane for angle calculation
        :return:
        '''
        d = q * forward
        y = d[..., 'xyz'.index(plane[0])]
        x = d[..., 'xyz'.index(plane[1])]
        return np.arctan2(y, x)