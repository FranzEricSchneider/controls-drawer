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

minLineLength = 150
lines = cv2.HoughLinesP(image=edges,
                        rho=0.02,
                        theta=np.pi/500,
                        threshold=10,
                        lines=np.array([]),
                        minLineLength=minLineLength,
                        maxLineGap=100)

a,b,c = lines.shape
print("Number of lines: {}".format(a))
for i in xrange(a):
    cv2.line(img=drawImg,
             pt1=(lines[i][0][0], lines[i][0][1]),
             pt2=(lines[i][0][2], lines[i][0][3]),
             color=(0, 0, 255),
             thickness=1,
             lineType=cv2.LINE_AA)

cv2.imshow('start', img)
cv2.imshow('edges', edges)
cv2.imshow('result', drawImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
