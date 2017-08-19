import argparse
import cv2
import lcm
import matplotlib.pyplot as plt
import time

from utils import lcm_msgs


class DisplayImages():
    def __init__(self, args):
        self.lcmobj = lcm.LCM()
        self.lcmobj.subscribe(args.image_channel, self.onImage)
        self.frame = None
        self.newFrame = False

    def run(self):
        while True:
            self.lcmobj.handle()

    def onImage(self, channel, data):
        # Set up window
        cv2.namedWindow(channel, cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()
        # Decode/parse out the image
        msg = lcm_msgs.auto_decode(channel, data)
        self.frame = lcm_msgs.image_t_to_nparray(msg)
        self.newFrame = True
        # Display the resulting frame
        cv2.imshow(channel, self.frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a basic image displayer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--image-channel",
                        help="Channel on which to track line position",
                        default="IMAGE_RAW")
    args = parser.parse_args()

    DI = DisplayImages(args)
    DI.run()
