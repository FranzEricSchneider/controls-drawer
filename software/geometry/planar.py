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
