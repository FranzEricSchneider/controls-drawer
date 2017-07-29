import numpy as np

from utils.geometry_tools import checkOrthonormal


def pixelsToImFrame(pixelPoint, calibMatrix):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if pixelPoint.shape != (2,) and pixelPoint.shape != (3,):
        raise ValueError("pixelsToImFrame accepts (2,) or (3,) arrays, not"
                         " {}".format(pixelPoint))
    assert calibMatrix.shape == (3, 3)
    if len(pixelPoint) == 2:
        pixelPoint = np.hstack((pixelPoint, 1))
    return np.linalg.inv(calibMatrix).dot(pixelPoint)


def imFrameToPixels(imFramePoint, calibMatrix):
    pass
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if imFramePoint.shape != (3,):
        raise ValueError("imFrameToPixels accepts (3,) arrays, not"
                         " {}".format(imFramePoint))
    assert calibMatrix.shape == (3, 3)
    return calibMatrix.dot(imFramePoint)[0:2]


def globalToPixels(point, HT, calibMatrix):
    checkOrthonormal(HT)

    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if point.shape == (3,) or point.shape == (4,):
        point = point.reshape((len(point), 1))
    elif point.shape != (3,1) and point.shape != (4,1):
        raise ValueError("globalToPixels accepts (3,1) or (4,1) arrays, not"
                         " {}".format(point))
    if len(point) == 3:
        point = np.vstack((point, 1))
    cameraFramePt = np.linalg.inv(HT).dot(point)
    unscaledPixels = calibMatrix.dot(cameraFramePt[0:3])
    unitPixels = unscaledPixels / unscaledPixels[2]
    return unitPixels[0:2, 0]
