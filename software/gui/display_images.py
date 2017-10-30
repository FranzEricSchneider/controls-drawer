import argparse
import cv2
import Queue
import signal
import sys
import threading
import time

import lcm

from utils import lcm_msgs


class DisplayImages():
    def __init__(self, args):
        print "KILL THE GUI WITH CNTL-C FROM TERMINAL"
        print "Otherwise you'll need cntl-z"
        # Set up LCM stuff
        self.lcmobj = lcm.LCM()
        self.lcmobj.subscribe(args.image_channel, self.onImage)
        if args.include_points:
            self.lcmobj.subscribe('IMAGE_POINTS_OF_INTEREST', self.onPoints)
        # Set up thread stuff
        self.points = None
        self.queue = Queue.Queue()
        self.flag = threading.Event()
        signal.signal(signal.SIGINT, self.signalHandler)
        self.displayThread = threading.Thread(
            target=displayImage, args=(self.queue, self.flag)
        )

    def run(self):
        self.displayThread.start()
        while True:
            lcm_msgs.lcmobj_handle_msg(self.lcmobj)

    def signalHandler(self, sentSignal, sentFrame):
        self.flag.set()
        sys.exit(0)

    def onImage(self, channel, data):
        image = lcm_msgs.image_t_to_nparray(lcm_msgs.auto_decode(channel, data))
        if self.points is not None:
            for point in self.points:
                pixel = tuple([int(x) for x in point])
                cv2.circle(image, pixel, radius=15, thickness=2, color=127)
        self.queue.put(image)

    def onPoints(self, channel, data):
        msg = lcm_msgs.auto_decode(channel, data)
        self.points = [[msg.axis_1[i], msg.axis_2[i]]
                       for i in range(msg.num_points)]


def displayImage(queue, flag):
    imageName = threading.currentThread().getName()
    frame = None
    while True:
        if queue.empty():
            time.sleep(0.01)
        else:
            if frame is None:
                cv2.namedWindow(imageName, cv2.WINDOW_AUTOSIZE)
                cv2.startWindowThread()
            frame = queue.get()
            cv2.imshow(imageName, frame)
            cv2.waitKey(1)
        if flag.isSet():
            cv2.destroyAllWindows()
            break
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a basic image displayer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--image-channel",
                        help="Channel on which to track line position",
                        default="IMAGE_RAW")
    parser.add_argument("-p", "--include-points",
                        help="Boolean, includes points of interest",
                        action="store_true")
    args = parser.parse_args()

    DI = DisplayImages(args)
    DI.run()
