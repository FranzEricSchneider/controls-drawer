import cv2

def writeImage(frame, path):
    cv2.imwrite(path, frame)
