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


def plotAxes(axes, HT):
    T = HT[0:3, 3]

    # X Axis
    axes.plot(xs=[T[0], T[0] + HT[0, 0]],
              ys=[T[1], T[1] + HT[1, 0]],
              zs=[T[2], T[2] + HT[2, 0]], color=(1.0, 0, 0))
    # Y Axis
    axes.plot(xs=[T[0], T[0] + HT[0, 1]],
              ys=[T[1], T[1] + HT[1, 1]],
              zs=[T[2], T[2] + HT[2, 1]], color=(0, 1.0, 0))
    # Z Axis
    axes.plot(xs=[T[0], T[0] + HT[0, 2]],
              ys=[T[1], T[1] + HT[1, 2]],
              zs=[T[2], T[2] + HT[2, 2]], color=(0, 0, 1.0))
