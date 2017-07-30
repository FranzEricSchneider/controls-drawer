import numpy as np

from geometry.cameras import globalToPixels, pixelsToGlobalPlane


def maskAroundToolframe(shape, HT, calibMatrix, radius=0.03):
    """
    Returns an mask with shape `shape ` that zeroes out an image out except for
    a radius around the (0, 0, 0) toolframe point in global space
    """
    mask = np.zeros(shape, dtype="bool")
    invCalibMatrix = np.linalg.inv(calibMatrix)
    for i in xrange(mask.shape[0]):
        for j in xrange(mask.shape[1]):
            pixel = np.array([i, j, 1.0])
            point = pixelsToGlobalPlane(pixel, HT, invCalibMatrix)
            # Since we're checking against (0, 0, 0) we can just take the
            # norm without getting a vector difference
            if any(point[0:2] > radius) or any(point[0:2] < -radius):
                # Skip the norm call below
                pass
            elif np.linalg.norm(point[0:2]) < radius:
                mask[i, j] = True
    return mask
