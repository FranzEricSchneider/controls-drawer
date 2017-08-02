import numpy as np
import pickle

from utils.geometry_tools import checkOrthonormal
from utils import navigation


class Camera():
    """
    General purpose camera class that will store geometrical information about
    the camera for easy reference
    """
    def __init__(self):
        intrinsicCalibResults = pickle.load(
            open(navigation.getLatestIntrinsicCalibration(), "rb"))
        exteriorCalibResults = pickle.load(
            open(navigation.getLatestExteriorCalibration(), "rb"))
        self.calibMatrix = intrinsicCalibResults['matrix']
        # Remember directionality:
        #   HT.dot(imFrame) = globalFrame
        #   inv(HT).dot(globalFrame) = imFrame
        self.HT = exteriorCalibResults['HT']


def pixelsToImFrame(pixelPoint, calibMatrix=None, invCalibMatrix=None):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if invCalibMatrix is None:
        invCalibMatrix = np.linalg.inv(calibMatrix)
    if pixelPoint.shape[0] != 2 and pixelPoint.shape[0] != 3:
        raise ValueError("pixelsToImFrame accepts (2,:) or (3,:) arrays, not"
                         " {}".format(pixelPoint.shape))
    if pixelPoint.shape[0] == 2 and len(pixelPoint.shape) == 1:
        pixelPoint = np.hstack((pixelPoint, 1))
    elif pixelPoint.shape[0] == 2:
        pixelPoint = np.vstack((pixelPoint, np.ones(1, pixelPoint.shape[1])))
    return invCalibMatrix.dot(pixelPoint)


def imFrameToPixels(imFramePoint, calibMatrix):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if imFramePoint.shape != (3,):
        raise ValueError("imFrameToPixels accepts (3,) arrays, not"
                         " {}".format(imFramePoint))
    unscaledPixels = calibMatrix.dot(imFramePoint)
    unitPixels = unscaledPixels / unscaledPixels[2]
    return unitPixels[0:2]


def globalToPixels(point, calibMatrix, HT=None, invHT=None):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if invHT is None:
        invHT = np.linalg.inv(HT)
    if point.shape == (3,) or point.shape == (4,):
        point = point.reshape((len(point), 1))
    elif point.shape != (3,1) and point.shape != (4,1):
        raise ValueError("globalToPixels accepts (3,1) or (4,1) arrays, not"
                         " {}".format(point.shape))
    if len(point) == 3:
        point = np.vstack((point, 1))
    print("invHT: {}".format(invHT))
    print("point: {}".format(point))
    imFramePoint = invHT.dot(point)
    return imFrameToPixels(imFramePoint[0:3, 0], calibMatrix)


def pixelsToGlobalPlane(pixelPoint, HT, invCalibMatrix):
    """
    Assuming the global plane is at z=0, project pixel points onto that plane.
    We can't otherwise write pixelsToGlobal because 2D doesn't contain enough
    information
    Accepts either a single point shape=(3,), or a set of points, shape=(3,n)
    """
    imFramePoints = pixelsToImFrame(pixelPoint, invCalibMatrix=invCalibMatrix)
    camOrigin = HT[0:3, 3]
    # By only using the rotation matrix we make this a vector, not a point
    unscaledGlobalVectors = HT[0:3, 0:3].dot(imFramePoints)
    # At this point we can calculate the correct scaling by setting the
    # resulting z value equal to 0. If we want to project on a non-(0,0,1)
    # plane then this will get more complicated
    # 0 = camOrigin[2] + k * globalVec[2]
    # k = -camOrigin[2] / globalVec[2]
    if len(unscaledGlobalVectors) == 1:
        scalar = -camOrigin[2] / unscaledGlobalVectors[2]
    else:
        scalar = -camOrigin[2] / unscaledGlobalVectors[2, :]
        camOrigin = camOrigin.reshape((3, 1))
    return camOrigin + scalar * unscaledGlobalVectors
