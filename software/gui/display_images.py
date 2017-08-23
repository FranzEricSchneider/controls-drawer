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
        # Set up LCM stuff
        self.lcmobj = lcm.LCM()
        self.lcmobj.subscribe(args.image_channel, self.onImage)
        # Set up thread stuff
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
        self.queue.put(
            lcm_msgs.image_t_to_nparray(lcm_msgs.auto_decode(channel, data))
        )


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
    args = parser.parse_args()

    DI = DisplayImages(args)
    DI.run()
