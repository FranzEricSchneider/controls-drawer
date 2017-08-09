import numpy as np

from geometry.cameras import globalToPixels, pixelsToGlobalPlane


def getCircularMask(shape, HT, calibMatrix, radius=0.03):
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


def getRingMask(shape, HT, calibMatrix, radius1, radius2):
    mask1 = getCircularMask(shape, HT, calibMatrix, radius1)
    mask2 = getCircularMask(shape, HT, calibMatrix, radius2)
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
    from geometry.cameras import Camera

    camera = Camera()
    frame = cv2.imread("/home/eon-alone/projects/controls-drawer/results/calibration_images/frames_1500083160330210/frame_SL15_X3_Y1_1500086784094601.png", 0)
    mask9mm = getCircularMask(frame.shape, camera.HT, camera.calibMatrix, 0.009)
    ring6to12 = getRingMask(frame.shape, camera.HT, camera.calibMatrix, 0.006, 0.012)

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    threshold1 = 50
    threshold2 = 200
    edges = cv2.Canny(frame, threshold1, threshold2)
    dilateEdges = cv2.dilate(edges, kernel, iterations=1)
    ring6to12Edges = dilateEdges * ring6to12

    circleEdges9 = cv2.Canny(mask9mm.astype(np.uint8) * 255, threshold1, threshold2)
    dilateCircleEdges9 = cv2.dilate(circleEdges9, kernel, iterations=1)
    intersection = (ring6to12Edges > 0) * (dilateCircleEdges9 > 0)
    # cv2.imwrite("ring6to12Edges.png", ring6to12Edges)
    # cv2.imwrite("dilateCircleEdges9.png", dilateCircleEdges9)
    # cv2.imwrite("intersection.png", intersection.astype(np.uint8) * 255)

    # cv2.imwrite("mask9mm.png", mask9mm.astype('double') * 255)
    # cv2.imwrite("ring6to12.png", ring6to12.astype('double') * 255)

    intersectionImage = intersection.astype(np.uint8) * 255
    intersectionContours, contours, hierarchy = \
        cv2.findContours(intersectionImage.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(intersectionContours, contours, -1, (120), 3)

    # print("intersectionContours: {}".format(intersectionContours))
    # print("contours: {}".format(contours))
    # print("hierarchy: {}".format(hierarchy))

    # cv2.imwrite("intersectionImage.png", intersectionImage)
    # cv2.imwrite("intersectionContours.png", intersectionContours)

    invCalibMatrix = np.linalg.inv(camera.calibMatrix)
    patchCenters = calcContourCenters(contours)
    pairs = findPairsOnLineEdge(patchCenters, camera.HT, invCalibMatrix)
    finalGlobalPoint = calcFinalGlobalPoint(pairs, patchCenters, camera.HT, invCalibMatrix, np.array([0.01, 0, 0]))
    print(pairs)
    print(patchCenters)
    print(finalGlobalPoint)

    finalPixel= np.round(globalToPixels(finalGlobalPoint, camera.calibMatrix, HT=camera.HT))
    print(finalPixel)
    center = tuple([int(x) for x in finalPixel])
    frame *= np.logical_not(dilateCircleEdges9.astype('bool'))
    cv2.circle(frame, center, radius=10, thickness=1, color=0)
    cv2.imwrite("framePlus.png", frame)
