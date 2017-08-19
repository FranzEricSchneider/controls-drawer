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
        self.displayMilliseconds = args.display_milliseconds

    def run(self):
        while True:
            self.lcmobj.handle()

    def onImage(self, channel, data):
        # Set up window
        cv2.namedWindow(channel, cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()
        # Decode/parse out the image
        msg = lcm_msgs.auto_decode(channel, data)
        frame = lcm_msgs.image_t_to_nparray(msg)
        # Display the resulting frame
        cv2.imshow(channel, frame)
        cv2.waitKey(self.displayMilliseconds)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a basic image displayer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--display-milliseconds",
                        help="How long to display each image",
                        type=int,
                        default=1000)
    parser.add_argument("-i", "--image-channel",
                        help="Channel on which to track line position",
                        default="IMAGE_RAW")
    args = parser.parse_args()

    DI = DisplayImages(args)
    DI.run()
