import cv2
import lcm
import numpy as np
from utils import lcm_msgs

def main():
    lc = lcm.LCM()
    lc.subscribe("IMAGE_CROPPED", displayImage)

    ch = "REQUEST_IMAGE"
    msg = lcm_msgs.auto_instantiate(ch)
    msg.format = msg.FORMAT_BGR
    # msg.format = msg.FORMAT_GRAY
    msg.n_arguments = 1
    msg.arg_names.append("crop")
    # msg.arg_values.append("0,0,863,479")
    msg.arg_values.append("100,10,663,300")
    msg.dest_channel = "IMAGE_CROPPED"
    lc.publish(ch, msg.encode())

    lc.handle()

def displayImage(channel, data):
    msg = lcm_msgs.auto_decode(channel, data)
    frame = lcm_msgs.image_t_to_nparray(msg)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()