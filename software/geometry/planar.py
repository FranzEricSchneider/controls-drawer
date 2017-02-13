################################################################################
# Does 2D geometry in the plane of the drawer. Generally takes 2D vectors
# (x, y, 0), 2D points (x, y, 1), or angles in radians
################################################################################

import numpy as np


def Rz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0,  1]])


def polygonVectors(numSides):
    """
    Takes a number of sides (e.g. 5) and returns a 3 x numSides array with the
    unit vectors for each side. The calling code can then scale the sides and
    place the origin where desired, and calculate the actual polygon points by
    adding the vectors up sequentially. This polygon has its first side on the
    X axis (1, 0, 0)
    """
    # Container for the eventual vectors
    vectors = []
    # Calculate the angle between each vector
    cornerAngle = 2 * np.pi / numSides
    R = Rz(cornerAngle)
    # Get a starting vector
    vectors.append(np.array([1, 0, 0]))
    # Loop until all vectors are made
    for i in xrange(1, numSides):
        vectors.append(R.dot(vectors[-1]))
    # Return a numpy array of the unit vectors
    return np.array(vectors)


def hatchLine(vector, numHatches):
    """
    Given a vector and a number of hatches to make this will return an array of
    points and a vector, so each hatch is of the type "go to point, draw the
    vector". The points and hatches will be defined from the origin of the
    original vector, so the calling code should add the vectors appropriately.
    The hatch length and spacing will be a fraction of the total vector length
    """
    # Calculate constants that will be used to find hatch points
    vectorLen = np.linalg.norm(vector)
    unitVector = vector / vectorLen
    hatchLen = 0.08 * vectorLen
    hatchSpace = 0.03 * vectorLen
    unitHatch = Rz(np.pi / 2.0).dot(unitVector)

    # Loop through the point indices and calculate each starting point
    points = []
    for i in xrange(numHatches):
        # Distance along the vector at which to draw the hatch
        dist = (vectorLen - hatchSpace * (numHatches - 1)) / 2.0 + (hatchSpace * i)
        points.append(unitVector * dist - unitHatch * (hatchLen / 2.0))

    # Return. Note that the vector will be the same for each hatch
    return np.array(points), unitHatch * hatchLen


class FiniteLine():
    def __init__(self, pts=None, pt1=None, pt2=None):
        if pts is None and pt1 is None and pt2 is None:
            # The basic line is a unit step along the x-axis
            self.pt1 = np.array([0.0, 0.0])
            self.pt2 = np.array([1.0, 0.0])
        elif pts is not None:
            self.pt1 = np.array([pts[0], pts[1]])
            self.pt2 = np.array([pts[2], pts[3]])
        elif pt1 is not None and pt2 is not None:
            self.pt1 = pt1
            self.pt2 = pt2
        else:
            raise ValueError('FiniteLine init called, unclear resolution')

    @property
    def x(self):
        return np.array([self.pt1[0], self.pt2[0]])

    @property
    def y(self):
        return np.array([self.pt1[1], self.pt2[1]])

    @property
    def intPt1(self):
        return np.array(self.pt1, dtype=np.int64)

    @property
    def intPt2(self):
        return np.array(self.pt2, dtype=np.int64)

    @property
    def length(self):
        return np.linalg.norm(self.pt2 - self.pt1)

    def getMidpoint(self, returnTupleInt=False):
        try:
            if returnTupleInt:
                return tuple([int(axis) for axis in self.midpoint])
            else:
                return self.midpoint
        except AttributeError:
            self.midpoint = np.array([np.average(self.x), np.average(self.y)])
            return self.getMidpoint(returnTupleInt)

    # An "average line" as I'm using it is formed by averaging the (x,y) values
    #   of the endpoints, where the chosen endpoints are the ones that will
    #   make the longest average line
    def average(self, line):
        if not isinstance(line, FiniteLine):
            # For now. Maybe in the future allow averaging with other lines
            raise ValueError('Muse average with another FiniteLine')
        lpt1 = line.pt1.copy()
        lpt2 = line.pt2.copy()
        # Check - would the endpoint pairs be shorter if we switched one of the
        #   lines, or are we best off if we leave the lines as is?
        pt1 = np.average([self.pt1, lpt1], axis=0)
        pt2 = np.average([self.pt2, lpt2], axis=0)
        switchPt1 = np.average([self.pt1, lpt2], axis=0)
        switchPt2 = np.average([self.pt2, lpt1], axis=0)
        if np.linalg.norm(switchPt2 - switchPt1) > np.linalg.norm(pt2 - pt1):
            return FiniteLine(pt1=switchPt1, pt2=switchPt2)
        else:
            return FiniteLine(pt1=pt1, pt2=pt2)

    def onImage(self, image, color=(0, 0, 255), thickness=2):
        # Color is in BGR
        import cv2
        cv2.line(img=image, pt1=tuple(self.intPt1), pt2=tuple(self.intPt2),
                 color=color, thickness=thickness)

    def plot(self, axis, color=(1.0, 0.0, 0.0), thickness=2):
        axis.plot(self.x, self.y, color=color, linewidth=thickness)


class InfiniteLine():
    def __init__(self, pt1=None, pt2=None, normal=None, bias=None, fLine=None):
        """
        Inputs:
            pt1/2 - Either 2x1 or 3x1 nparrays
            normal - 2x1 unit vector
            bias - float of how far from the origin the closest point is

        There are two ways to represent lines here:
          1) If given two 2D points, we can reconstruct the line by taking
             the different between the points
          2) If given a normal and a vector the line is defined as being
             perpendicular to the normal and bias units from the origin
        """
        if (pt1 is None and pt2 is None and normal is None and bias is None
            and fLine is None):
            # The basic line is along the X axis, intersecting the origin
            self.normal = np.array([0.0, 1.0])
            self.bias = 0.0
        elif pt1 is not None and pt2 is not None:
            lineVector = pt2 - pt1
            # By crossing a vector in the 2D (x, y) plane with a z vector we
            #   get a vector perpendicular to lineVector, not much caring about
            #   any particular (+-) orientation
            if lineVector.shape == (2,):
                self.normal = np.cross(np.hstack((lineVector, 0.0)),
                                       np.array([0.0, 0.0, 1.0]))[:2]
            else:
                self.normal = np.cross(lineVector,
                                       np.array([0.0, 0.0, 1.0]))[:2]
            self.normal /= np.linalg.norm(self.normal)

            # Calculate the bias from normal to line. The equation we will use
            #   is thus: if we take the unit vector between the two points,
            #   scale it by a value k, and add it to one of the points, then
            #   at some point it will intersect the normal vector, extended
            #   from the origin with a scaling factor of b (bias). If v is the
            #   point to point vector and n is the normal, we have
            #       x + k * v1 = b * n1
            #       y + k * v2 = b * n2
            #   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            #       x = M * k   M = v1, n1   Note: because v direction is
            #       y       b       v2, n2         arbitary we can negate v
            alongLine = lineVector / np.linalg.norm(lineVector)
            # This matrix is orthonormal because normal is orthogonal to
            #   alongLine by definition and they are unit vectors
            M = np.array([[alongLine[0], self.normal[0]],
                          [alongLine[1], self.normal[1]]])
            kb = M.T.dot(pt1)
            self.bias = kb[1]
            if self.bias < 0:
                self.normal *= -1
                self.bias *= -1

        elif normal is not None and bias is not None:
            self.normal = normal
            self.bias = bias
        elif fLine is not None:
            self.__init__(pt1=fLine.pt1, pt2=fLine.pt2)
        else:
            raise ValueError('InfiniteLine init called, unclear resolution')

    @property
    def parallel(self):
        try:
            return self.parallelLine
        except AttributeError:
            self.parallelLine = np.cross(np.hstack((self.normal, 0.0)),
                                         np.array([0.0, 0.0, 1.0]))[:2]
            return self.parallelLine

    def onImage(self, image, color=(0, 0, 255), thickness=2):
        # Color is in BGR
        import cv2
        # Make the length of the vectors definitely longer that the picture,
        #   because the normals are not garuanteed to originate from the image
        length = np.max(image.shape) * 2
        midpoint = self.normal * self.bias
        endPoints = np.array([midpoint + self.parallel * length,
                              midpoint - self.parallel * length], dtype=np.int64)
        cv2.line(img=image, pt1=tuple(endPoints[0, :]), pt2=tuple(endPoints[1, :]),
                 color=color, thickness=thickness)

    def plot(self, axis, length=10, color=(1.0, 0.0, 0.0), thickness=2):
        midpoint = self.normal * self.bias
        endPoints = np.array([midpoint + self.parallel * length,
                              midpoint - self.parallel * length])
        axis.plot(
            endPoints[:, 0], endPoints[:, 1], color=color, linewidth=thickness
        )
