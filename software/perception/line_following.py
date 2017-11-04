import cv2
import numpy as np

from geometry.cameras import globalToPixels
from geometry.cameras import pixelsToGlobalPlane
from geometry.cameras import getMaskBounds
from geometry.cameras import cropImage
from utils.lcm_msgs import auto_decode
from utils.lcm_msgs import image_t_to_nparray


def getCircularMask(shape, HT, invCalibMatrix, radius=0.03):
    """
    Returns an mask with shape `shape ` that zeroes out an image out except for
    a radius around the (0, 0, 0) toolframe point in global space
    """
    pixelCoords = np.array([[j, i, 1.0]
                            for i in xrange(shape[0])
                            for j in xrange(shape[1])]).T

    pointsXY = pixelsToGlobalPlane(pixelCoords, HT, invCalibMatrix)[0:2].T
    # First (fast check) remove all points with one axis greater than radius
    mask = np.all(abs(pointsXY) < radius, axis=1)
    # For the remainder, make sure the norm is less than radius
    mask[mask] = np.linalg.norm(pointsXY[mask], axis=1) < radius
    return mask.reshape((shape))


def getRingMask(shape, HT, invCalibMatrix, radius1, radius2):
    mask1 = getCircularMask(shape, HT, invCalibMatrix, radius1)
    mask2 = getCircularMask(shape, HT, invCalibMatrix, radius2)
    return np.logical_xor(mask1, mask2)


def calcContourCenters(contours):
    """
    Inputs
        contours: output from the cv2.findContours function, which is a list of
            (n, 1, 2) arrays, each listing the vertices of a particular patch
    """
    patchCenters = []
    for patch in contours:
        patchCenters.append(np.round(np.average(patch, axis=0)[0]))
    return patchCenters


def findPairsOnLineEdge(patchCenters, HT, invCalibMatrix, width=0.001, allowedError=0.002):
    """
    Find pairs of contours within a certain distance in global terms

    Inputs
        patchCenters: the pixel center of each point of interest
        width: approximate width of the line in meters
        allowedError: allowed error between distance and width to count as pair
    Outputs
        pairs:
        patchCenters: A list with the centered pi
    """
    numPatches = len(patchCenters)

    # Find the index of the other contour closest to the given width away
    hasMate = [None] * numPatches
    for i, patchCenter1 in enumerate(patchCenters):
        for j, patchCenter2 in enumerate(patchCenters):
            if i == j:
                continue

            # TODO: Consider the speed aspect and don't recalculate the pixel
            #       to global relationship n^2 times when you could do it n
            #       times

            point1 = pixelsToGlobalPlane(patchCenter1, HT, invCalibMatrix)
            point2 = pixelsToGlobalPlane(patchCenter2, HT, invCalibMatrix)
            distance = np.linalg.norm(point1 - point2)
            error = abs(distance - width)
            isMate = error < allowedError
            if isMate:
                if hasMate[i] is None or error < hasMate[i][1]:
                    hasMate[i] = (j, error)

    # Pick the pairs of contours that both have each other as the best bet
    accountedIndices = []
    pairs = []
    for i in xrange(numPatches):
        if i in accountedIndices:
            continue
        elif hasMate[i] is None:
            continue
        elif hasMate[hasMate[i][0]][0] == i:
            pairs.append((i, hasMate[i][0]))
            accountedIndices.append(hasMate[i][0])

    return pairs


def calcFinalGlobalPoint(pairs, patchCenters, HT, invCalibMatrix, lastPoint):
    """
    Returns the global point (pixel points formed by averaging the pixel pairs)
    that is closest to the previous point, and None if no valid points are
    found

    Inputs
        lastPoint: The position in meters of the last known global point
    """
    # TODO: Test for speed and consider speeding this up by instead converting
    #       lastPoint into pixel coordinate and comparing pixel differences
    points = [
        np.average(np.vstack(
            (pixelsToGlobalPlane(patchCenters[i], HT, invCalibMatrix),
             pixelsToGlobalPlane(patchCenters[j], HT, invCalibMatrix))
        ), axis=0)
        for i, j in pairs
    ]
    distances = [np.linalg.norm(lastPoint - point) for point in points]
    try:
        return points[distances.index(min(distances))]
    except ValueError:
        # If distances is [] then we come here
        return None


def findPointInFrame(frame, bounds, kernel, ringOfInterest, HT,
                     invCroppedCalibMatrix, pastGlobalPoint, width=0.003,
                     threshold1=70, threshold2=180):
    """
    TODO:

    Inputs:
        TODO
    """
    # Find edges in cropped frame
    edges = cv2.Canny(cropImage(frame, bounds), threshold1, threshold2)
    dilateEdges = cv2.dilate(edges, kernel, iterations=1)    

    # Find contours where the image edges cross the ring of interest
    intersection = (dilateEdges > 0) * (ringOfInterest > 0)
    intersectionContours, contours, hierarchy = \
        cv2.findContours(intersection.astype(np.uint8) * 255,
                         cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Take contour pairs and find which one best matches up with the past point
    patchCenters = calcContourCenters(contours)
    pairs = findPairsOnLineEdge(patchCenters, HT, invCroppedCalibMatrix,
                                width=width)
    return calcFinalGlobalPoint(pairs, patchCenters, HT, invCroppedCalibMatrix,
                                pastGlobalPoint)


if __name__ == '__main__':
    import cv2
    import glob
    from geometry.cameras import Camera

    camera = Camera()
    frameShape = (460, 621)
    mask9mm = getCircularMask(frameShape, camera.HT, camera.invCalibMatrix, 0.009)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    threshold1 = 70
    threshold2 = 180
    circleEdges9 = cv2.Canny(mask9mm.astype(np.uint8) * 255, threshold1, threshold2)
    dilateCircleEdges9 = cv2.dilate(circleEdges9, kernel, iterations=1)
    width = 0.002

    # frame = cv2.imread("/home/eon-alone/projects/controls-drawer/results/calibration_images/frames_1500083160330210/frame_SL15_X3_Y1_1500086784094601.png", 0)
    # frameNames = sorted(glob.glob("/home/eon-alone/projects/controls-drawer/results/line_following/test_8_10/post_threshold/frame*.png"))
    # frameNames = sorted(glob.glob("/home/eon-alone/projects/controls-drawer/results/line_following/test_8_10/frame*.png"))
    # frames = [cv2.imread(frameName, 0) for frameName in frameNames]

    import lcm
    good_utimes = [1509330216529368, 1509330216627868, 1509330216728825, 1509330216834332, 1509330216930756, 1509330217034101, 1509330217131165, 1509330217231357, 1509330217328597, 1509330217434614, 1509330217535127, 1509330217631806, 1509330217731755, 1509330217831779, 1509330217934868, 1509330218030021, 1509330218130679, 1509330218237284, 1509330218334279, 1509330218434172, 1509330218532731, 1509330218637267, 1509330218732012, 1509330218834588, 1509330218937336, 1509330219036061, 1509330219139660, 1509330219338095, 1509330219436881, 1509330219536097, 1509330219637484, 1509330219739777, 1509330219834151, 1509330219941932, 1509330220038227, 1509330220136414, 1509330220235189, 1509330220338863, 1509330220541411, 1509330220636081, 1509330221843285, 1509330221980431, 1509330223549131, 1509330223680151, 1509330223744502, 1509330223948095, 1509330224048316, 1509330224181165, 1509330224247384, 1509330224381555, 1509330224585291, 1509330224655548, 1509330224750438, 1509330224887206, 1509330224949192, 1509330225092424, 1509330225351838, 1509330225790952, 1509330226592358, 1509330226658098, 1509330226785008, 1509330226858883, 1509330226952589, 1509330227091694, 1509330227153385, 1509330227292016, 1509330227356853, 1509330227453889, 1509330227591879, 1509330227653898, 1509330227791592, 1509330227859639, 1509330229057569]
    log = lcm.EventLog("/home/eon-alone/projects/controls-drawer/results/line_following/test_10_29/lcmlog-2017-10-29_points_of_interest")
    channel = "IMAGE_RAW"
    frames = [image_t_to_nparray(auto_decode(event.channel, event.data))
              for event in log
              if (event.channel == channel and
                  auto_decode(event.channel, event.data).utime in good_utimes)]
    assert len(good_utimes) == len(frames)
    frameNames = ["{}".format(utime) for utime in good_utimes]

    finalGlobalPoint = np.array([0, 0.01, 0])

    # Set up cropped masks and calibration matrices so that we always crop to
    # the outer ring + some pixels
    maskBounds = getMaskBounds(mask9mm, pixBuffer=10)
    croppedDilatedEdges9, croppedCalibMatrix = \
        cropImage(dilateCircleEdges9, maskBounds, camera.calibMatrix)
    invCroppedCalib = np.linalg.inv(croppedCalibMatrix)

    for frame, frameName in zip(frames, frameNames):
        # Find the point
        foundPoint = findPointInFrame(frame=frame,
                                      bounds=maskBounds,
                                      kernel=kernel,
                                      ringOfInterest=croppedDilatedEdges9,
                                      HT=camera.HT,
                                      invCroppedCalibMatrix=invCroppedCalib,
                                      pastGlobalPoint=finalGlobalPoint,
                                      width=width)

        # Prepare a color frame for diplay later
        frame *= np.logical_not(dilateCircleEdges9.astype('bool'))
        colorFrame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

        if foundPoint is None:
            print("Picture {} hopeless! No overlap in the ring".format(frameName))
        else:
            finalGlobalPoint = foundPoint
            finalPixel = np.round(globalToPixels(finalGlobalPoint, camera.calibMatrix, HT=camera.HT))
            center = tuple([int(x) for x in finalPixel])
            cv2.circle(colorFrame, center, radius=15, thickness=2, color=(0, 0, 255))

        cv2.imshow('frame_plus_identified_points', colorFrame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
