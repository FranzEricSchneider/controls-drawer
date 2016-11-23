import cv2
import numpy as np
import time


cap = cv2.VideoCapture(1)
times = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    times.append(time.time())

meanTime = np.mean(np.diff(times))
print("time: {}, freq: {}".format(meanTime, 1 / meanTime))

# Possible known combinations
#	(1280, 720), (864, 480), (640, 480), (640, 360), (176, 144)
print("width: {}".format(cap.get(3)))
print("height: {}".format(cap.get(4)))
# The max value is 30, you set FPS values lower than 30 (rounds to nearest 5)
print("FPS: {}".format(cap.get(5)))
# The brightness was naturally around 0.5, and can be adjusted up and down to
# 	make the image brighter/darker. It's interesting because the camera will
# 	light balance a given scene without changing the brightness
print("Brightness: {}".format(cap.get(10)))
# The contrast looked normal around 0.125, especially when viewing it in color.
# 	When doing black and white pictures high contrast could be really helpful if
# 	the light sources are correct - it basically can give you clear images of
# 	the things of interest IF the system is set up in clear dark/light colors
# 	to display those things correctly already. If the system doesn't just work
# 	then you should do the filters by hand to get the things of interest
print("Contrast: {}".format(cap.get(11)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
