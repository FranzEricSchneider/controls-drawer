import numpy as np
import cv2

from geometry.planar import FiniteLine, InfiniteLine


img = cv2.imread('frame_SL15_X-10_Y15_1486332586617381.png')
# allLines = [[0, 3], [10, 11], [1, 6], [4, 12], [9, 2]]

drawImg = img.copy()

threshold1 = 90
threshold2 = 100
edges = cv2.Canny(img, threshold1, threshold2)

minLineLength = 40
lines = cv2.HoughLinesP(image=edges,
                        rho=4,
                        theta=np.pi/180,
                        threshold=10,
                        lines=np.array([]),
                        minLineLength=minLineLength,
                        maxLineGap=20)

fLines = [FiniteLine(line[0]) for line in lines]
numLines = len(fLines)
print("Number of lines: {}".format(numLines))
for fLine in fLines:
    fLine.onImage(drawImg)

cv2.namedWindow("initial_detection", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
# cv2.imshow('start', img)
# cv2.imshow('edges', edges)
cv2.imshow('initial_detection', drawImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

result = ""
indices = range(len(fLines))
while result != "q":
    textImg = img.copy()

    # Try to remove the indicated index
    try:
        badIndices = [int(index.strip()) for index in result.split(" ")]
        for index in badIndices:
            indices.remove(index)
    except ValueError:
        pass

    for i in indices:
        fLines[i].onImage(textImg)
        midpoint = fLines[i].getMidpoint(returnTupleInt=True)
        cv2.putText(textImg, "{}".format(i), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    print("Look at the image and decide which line numbers you can delete")
    cv2.namedWindow("text", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    cv2.imshow('text', textImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = raw_input('Which numbers would you delete? enter w/ spaces (e.g.'
                       ' 12 0 5[enter]) or hit q to move on: ')

print("Moving on! indices: {}".format(indices))

# Now we have an image and just the lines that correspond to the pentagon. Time
#   to map the lines to specific edges of the pentagon
allLines = []
for i in xrange(5):
    print("Look - which lines corresponds to edge {}?".format(i))
    cv2.namedWindow("text", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    cv2.imshow('text', textImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = raw_input("Which two lines correspond to edge {}? Enter with"
                       " spaces in between (e.g. 11 3): ".format(i))
    singleEdgeLines = [int(index.strip()) for index in result.split(" ")]
    allLines.append(singleEdgeLines)
print("allLines! {}".format(allLines))

# Time to estimate midlines and points
midLines = [fLines[allLines[index][0]].average(fLines[allLines[index][1]])
            for index in xrange(5)]
midlineImage = img.copy()
for line in midLines:
    line.onImage(midlineImage, color=(0, 255, 0), thickness=2)

cv2.namedWindow("midlines", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow('midlines', midlineImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
