import numpy as np


def checkOrthonormal(HT):
    assert isOrthonormal(HT)


def isOrthonormal(HT):
    xVector = HT[0:3, 0]
    yVector = HT[0:3, 1]
    zVector = HT[0:3, 2]

    # Check for unit vectorness
    if not np.isclose(np.linalg.norm(xVector), 1.0):
        return False
    if not np.isclose(np.linalg.norm(yVector), 1.0):
        return False
    if not np.isclose(np.linalg.norm(zVector), 1.0):
        return False

    # Check that certain values are just set to 0 or 1
    for zero in HT[3, 0:3]:
        if zero != 0:
            return False
    if HT[3, 3] != 1:
        return False

    # Check for orthonormality
    for a, b in zip([xVector, yVector, zVector], [yVector, zVector, xVector]):
        if not np.isclose(a.dot(b), 0.0):
            return False

    # Check that cross(x, y) = z relationship exists
    if not np.isclose(np.cross(xVector, yVector).dot(zVector), 1.0):
        return False

    return True
