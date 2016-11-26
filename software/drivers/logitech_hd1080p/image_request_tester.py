import cv2
import lcm
import numpy as np
from utils import lcm_msgs

def main():
    lc = lcm.LCM()
    lc.subscribe("IMAGE_RAW", displayImage)

    ch = "REQUEST_IMAGE"
    msg = lcm_msgs.auto_instantiate(ch)
    msg.format = msg.FORMAT_BGR
    lc.publish(ch, msg.encode())

    lc.handle()

def displayImage(channel, data):
    msg = lcm_msgs.auto_decode(channel, data)
    frame = np.array([datum for datum in msg.data], dtype=np.uint8)
    if msg.request.format == msg.request.FORMAT_GRAY:
        frame = frame.reshape(msg.height, msg.width)
    elif msg.request.format == msg.request.FORMAT_BGR:
        frame = frame.reshape(msg.height, msg.width, 3)
    else:
        raise Exception

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()