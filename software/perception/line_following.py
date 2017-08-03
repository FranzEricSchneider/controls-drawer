import numpy as np

from geometry.cameras import globalToPixels, pixelsToGlobalPlane


def maskAroundToolframe(shape, HT, calibMatrix, radius=0.03):
    """
    Returns an mask with shape `shape ` that zeroes out an image out except for
    a radius around the (0, 0, 0) toolframe point in global space
    """
    invCalibMatrix = np.linalg.inv(calibMatrix)
    pixelCoords = np.array([[j, i, 1.0]
                            for i in xrange(shape[0])
                            for j in xrange(shape[1])]).T

    pointsXY = pixelsToGlobalPlane(pixelCoords, HT, invCalibMatrix)[0:2].T
    # First (fast check) remove all points with one axis greater than radius
    mask = np.all(abs(pointsXY) < radius, axis=1)
    # For the remainder, make sure the norm is less than radius
    mask[mask] = np.linalg.norm(pointsXY[mask], axis=1) < radius
    return mask.reshape((shape))


if __name__ == '__main__':
    import cv2
    from geometry.cameras import Camera

    camera = Camera()
    frame = cv2.imread("/home/eon-alone/projects/controls-drawer/results/calibration_images/frames_1500083160330210/frame_SL15_X-2_Y10_1500086910844077.png", 0)
    mask6mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.006)
    mask9mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.009)
    mask12mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.012)
    ring6to9 = np.logical_xor(mask6mm, mask9mm)
    ring6to12 = np.logical_xor(mask6mm, mask12mm)
    ring9to12 = np.logical_xor(mask9mm, mask12mm)
    cv2.imwrite("mask6mm.png", mask6mm.astype('double') * 255)
    cv2.imwrite("mask9mm.png", mask9mm.astype('double') * 255)
    cv2.imwrite("mask12mm.png", mask12mm.astype('double') * 255)
    cv2.imwrite("ring6to9.png", ring6to9.astype('double') * 255)
    cv2.imwrite("ring6to12.png", ring6to12.astype('double') * 255)
    cv2.imwrite("ring9to12.png", ring9to12.astype('double') * 255)
