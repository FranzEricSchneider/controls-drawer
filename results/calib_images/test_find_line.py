import numpy as np
import cv2

from geometry.planar import FiniteLine, InfiniteLine
from utils.ui import AskAboutLinesToMerge
from utils.ui import AskAboutLinesToRemove
from utils.ui import AskAboutPentagonLines


img = cv2.imread('frame_SL15_X-10_Y10_1489807439296551.png')
allLines = [[0, 3], [10, 11], [1, 6], [4, 12], [9, 2]]

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

# Do you want to delete any?
indices = range(len(fLines))

# Remove lines
lineRemover = AskAboutLinesToRemove(img, fLines, indices)
lineRemover.processImage()
indices = lineRemover.getIndices()
fLines = lineRemover.getLines()

# Merge lines
lineMerger = AskAboutLinesToMerge(img, fLines, indices)
lineMerger.processImage()
indices = lineMerger.getIndices()
fLines = lineMerger.getLines()

# Now we have an image and just the lines that correspond to the pentagon. Time
#   to map the lines to specific edges of the pentagon
allLines = []
pentagon = AskAboutPentagonLines(img, fLines, indices)
pentagon.processImage()
indicesBySide = pentagon.getIndicesBySide()
for i in xrange(5):
    allLines.append(indicesBySide[i])
print("allLines! {}".format(allLines))


# Time to estimate midlines and points
midLines = [fLines[allLines[index][0]].average(fLines[allLines[index][1]])
            for index in xrange(5)]
midlineImage = img.copy()
midILines = [InfiniteLine(fLine=line) for line in midLines]
for line in midILines:
    line.onImage(midlineImage, thickness=1)
for line in midLines:
    line.onImage(midlineImage, color=(0, 255, 0), thickness=2)
intIndices = np.array(range(len(midLines)))
for i, j in zip(intIndices, np.roll(intIndices, 1)):
    intersection = midILines[i].intersection(midILines[j])
    center = tuple([int(x) for x in intersection])
    cv2.circle(midlineImage, center, radius=6, color=(204, 255, 0), thickness=2)
    cv2.circle(midlineImage, center, radius=6, color=(0, 0, 0), thickness=1)

cv2.namedWindow("midlines", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow('midlines', midlineImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
