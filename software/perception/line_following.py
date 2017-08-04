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
    frame = cv2.imread("/home/eric/projects/controls-drawer/results/calibration_images/frames_1500083160330210/frame_SL15_X3_Y1_1500086784094601.png", 0)
    mask6mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.006)
    mask8mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.008)
    mask10mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.010)
    mask12mm = maskAroundToolframe(frame.shape, camera.HT, camera.calibMatrix, 0.012)
    ring8to10 = np.logical_xor(mask8mm, mask10mm)
    ring6to12 = np.logical_xor(mask6mm, mask12mm)

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    threshold1 = 50
    threshold2 = 200
    edges = cv2.Canny(frame, threshold1, threshold2)
    dilateEdges = cv2.dilate(edges, kernel, iterations=1)
    ring6to12Edges = dilateEdges * ring6to12

    ringEdges8to10 = cv2.Canny(ring8to10.astype(np.uint8) * 255, threshold1, threshold2)
    dilateRingEdges8to10 = cv2.dilate(ringEdges8to10, kernel, iterations=1)
    intersection = (ring6to12Edges > 0) * (dilateRingEdges8to10 > 0)
    cv2.imwrite("ring6to12Edges.png", ring6to12Edges)
    cv2.imwrite("dilateRingEdges8to10.png", dilateRingEdges8to10)
    cv2.imwrite("intersection.png", intersection.astype(np.uint8) * 255)

    # cv2.imwrite("mask6mm.png", mask6mm.astype('double') * 255)
    # cv2.imwrite("mask9mm.png", mask9mm.astype('double') * 255)
    # cv2.imwrite("mask12mm.png", mask12mm.astype('double') * 255)
    cv2.imwrite("ring8to10.png", ring8to10.astype('double') * 255)
    cv2.imwrite("ring6to12.png", ring6to12.astype('double') * 255)
