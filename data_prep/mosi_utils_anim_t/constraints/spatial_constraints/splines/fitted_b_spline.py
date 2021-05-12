import numpy as np
import scipy.interpolate as si
B_SPLINE_DEGREE = 3


class FittedBSpline(object):
    def __init__(self,  points, degree=B_SPLINE_DEGREE, domain=None):
        self.points = np.array(points)
        if isinstance(points[0], (int, float, complex)):
            self.dimensions = 1
        else:
            self.dimensions = len(points[0])
        self.degree = degree
        if domain is not None:
            self.domain = domain
        else:
            self.domain = (0.0, 1.0)

        self.initiated = True
        self.spline_def = []
        points_t = np.array(points).T
        t_func = np.linspace(self.domain[0], self.domain[1], len(points)).tolist()
        for d in range(len(points_t)):
            #print d, self.dimensions
            self.spline_def.append(si.splrep(t_func, points_t[d], w=None, k=3))

    def _initiate_control_points(self):
        return

    def clear(self):
        return

    def query_point_by_parameter(self, u):
        """

        """
        point = []
        for d in range(self.dimensions):
            point.append(si.splev(u, self.spline_def[d]))
        return np.array(point)

    def get_last_control_point(self):
        return self.points[-1]