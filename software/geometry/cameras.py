import numpy as np


def pixelsToImFrame(pixelPoint, calibMatrix):
    # TODO: Assumptions about point
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if len(pixelPoint) == 2:
        pixelPoint = np.hstack((pixelPoint, 1))
    return np.linalg.inv(calibMatrix).dot(pixelPoint)


def globalToPixels(point, HT, calibMatrix):
    # TODO: Assumptions about point
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if len(point) == 3:
        point = np.hstack((point, 1))
    cameraFramePt = HT.dot(point)
    unscaledPixels = calibMatrix.dot(cameraFramePt[0:3])
    unitPixels = unscaledPixels / unscaledPixels[2]
    return unitPixels[0:2]
