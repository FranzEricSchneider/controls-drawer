import numpy as np

from geometry.cameras import globalToPixels
from geometry.cameras import pixelsToGlobalPlane
from geometry.cameras import getMaskBounds
from geometry.cameras import cropImage


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
    that is closest to the previous point

    Inputs
        lastPoint: The position in meters of the last known global point
    """
    points = [
        np.average(np.vstack(
            (pixelsToGlobalPlane(patchCenters[i], HT, invCalibMatrix),
             pixelsToGlobalPlane(patchCenters[j], HT, invCalibMatrix))
        ), axis=0)
        for i, j in pairs
    ]
    distances = [np.linalg.norm(lastPoint - point) for point in points]
    return points[distances.index(min(distances))]


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


    # frame = cv2.imread("/home/eon-alone/projects/controls-drawer/results/calibration_images/frames_1500083160330210/frame_SL15_X3_Y1_1500086784094601.png", 0)
    # frameNames = sorted(glob.glob("/home/eon-alone/projects/controls-drawer/results/line_following/test_8_10/post_threshold/frame*.png"))
    frameNames = sorted(glob.glob("/home/eon-alone/projects/controls-drawer/results/line_following/test_8_10/frame*.png"))
    frames = [cv2.imread(frameName, 0) for frameName in frameNames]
    finalGlobalPoint = np.array([0, 0.01, 0])

    # Set up cropped masks and calibration matrices so that we always crop to
    # the outer ring + some pixels
    maskBounds = getMaskBounds(mask9mm, pixBuffer=10)
    croppedDilatedEdges9, croppedCalibMatrix = \
        cropImage(dilateCircleEdges9, maskBounds, camera.calibMatrix)
    invCroppedCalib = np.linalg.inv(croppedCalibMatrix)

    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    for frame in frames:
        cropFrame = cropImage(frame, maskBounds)
        edges = cv2.Canny(cropFrame, threshold1, threshold2)
        dilateEdges = cv2.dilate(edges, kernel, iterations=1)

        intersection = (dilateEdges > 0) * (croppedDilatedEdges9 > 0)

        intersectionImage = intersection.astype(np.uint8) * 255
        intersectionContours, contours, hierarchy = \
            cv2.findContours(intersectionImage.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(intersectionContours, contours, -1, (120), 3)

        hasContours = True
        if len(contours) == 0:
            # print("Picture {} hopeless! No overlap in the ring".format(frameName))
            hasContours = False

        frame *= np.logical_not(dilateCircleEdges9.astype('bool'))
        colorFrame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

        if hasContours:
            # cv2.imwrite("intersectionImage.png", intersectionImage)
            # cv2.imwrite("intersectionContours.png", intersectionContours)

            patchCenters = calcContourCenters(contours)
            pairs = findPairsOnLineEdge(patchCenters, camera.HT, invCroppedCalib, width=0.003)
            finalGlobalPoint = calcFinalGlobalPoint(pairs, patchCenters, camera.HT, invCroppedCalib, finalGlobalPoint)

            finalPixel = np.round(globalToPixels(finalGlobalPoint, camera.calibMatrix, HT=camera.HT))
            center = tuple([int(x) for x in finalPixel])
            cv2.circle(colorFrame, center, radius=15, thickness=2, color=(0, 0, 255))

        cv2.imshow('frame_plus_identified_points', colorFrame)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

    # pr.disable()
    # import time
    # now = int(time.time() * 1e6)
    # pr.dump_stats("stats_{}.runsnake".format(now))
