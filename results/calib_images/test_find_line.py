import numpy as np
import cv2

# img = cv2.imread('frame_SL15_X0_Y20_1486311755049016.png')
# img = cv2.imread('frame_SL15_X0_Y20_1486311832076731.png')
# img = cv2.imread('frame_SL15_X-10_Y15_1486332420908549.png')
# img = cv2.imread('frame_SL15_X-10_Y15_1486332518417497.png')
# img = cv2.imread('frame_SL15_X-10_Y15_1486332571440002.png')
img = cv2.imread('frame_SL15_X-10_Y15_1486332586617381.png')
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

a,b,c = lines.shape
print("Number of lines: {}".format(a))
for i in xrange(a):
    cv2.line(img=drawImg,
             pt1=(lines[i][0][0], lines[i][0][1]),
             pt2=(lines[i][0][2], lines[i][0][3]),
             color=(0, 0, 255),
             thickness=1)

cv2.namedWindow("initial_detection", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
# cv2.imshow('start', img)
# cv2.imshow('edges', edges)
cv2.imshow('initial_detection', drawImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

result = ""
indices = range(a)
while result != "q":
    textImg = img.copy()

    # Try to remove the indicated index
    try:
        badIndices = [int(index.strip()) for index in result.split(",")]
        for index in badIndices:
            indices.remove(index)
    except ValueError:
        pass

    for i in indices:
        cv2.line(img=textImg,
                 pt1=(lines[i][0][0], lines[i][0][1]),
                 pt2=(lines[i][0][2], lines[i][0][3]),
                 color=(0, 0, 255),
                 thickness=1)
        midPoint = np.average((lines[i][0][:2], lines[i][0][2:]), axis=0)
        midPoint = tuple([int(axis) for axis in midPoint])
        cv2.putText(textImg, "{}".format(i), midPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    print("Look at the image and decide which line numbers you can delete")
    cv2.namedWindow("text", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    cv2.imshow('text', textImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = raw_input('Which numbers would you like to delete?\n'
                       '\tenter as CSV (e.g. 12, 0, 5[enter])\n'
                       '\tor hit q to move on')

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
    result = raw_input("Which two lines correspond to edge {}? Enter as csv".format(i))
    singleEdgeLines = [int(index) for index in result.split(",")]
    allLines.append(singleEdgeLines)
print("allLines! {}".format(allLines))
