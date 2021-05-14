__author__ = 'erhe01'

class SplineSegment(object):
    def __init__(self, start, center, end):
        self.start = start
        self.center = center
        self.end = end
        return

    def divide(self):
        """Divides a segment into two segments
        Returns
        -------
        * segments : list of SplineSegments
            Contains segment_a and segment_b. Each defines a line segment and
            contains start, center and end points
        """
        center_a = 0.5 * (self.center - self.start) + self.start
        center_b = 0.5 * (self.end - self.center) + self.center
        return [SplineSegment(self.start, center_a, self.center), SplineSegment(self.center, center_b, self.end)]


