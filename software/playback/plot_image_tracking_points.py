import argparse
import cv2
import lcm

from utils.lcm_msgs import auto_decode
from utils.lcm_msgs import image_t_to_nparray


POINTS_CHANNEL = 'IMAGE_POINTS_OF_INTEREST'


def main(log, image_channel, frame_time):
    check_prerequisites(log, [image_channel, POINTS_CHANNEL])
    image_msgs = [auto_decode(image_channel, msg.data)
                  for msg in log
                  if msg.channel == image_channel]
    point_msgs = [auto_decode(POINTS_CHANNEL, msg.data)
              for msg in log
              if msg.channel == POINTS_CHANNEL]

    # Set up the image display window
    cv2.namedWindow(POINTS_CHANNEL, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    # Go through each image
    points = None
    for image_msg in image_msgs:
        image = image_t_to_nparray(image_msg)

        try:
            point_msg = point_msgs[0]
            # If the point was found before the image was created, update the
            # points variable and pop the point off the list
            if point_msg.utime < image_msg.utime:
                points = [tuple([int(point_msg.axis_1[i]),
                                 int(point_msg.axis_2[i])])
                          for i in range(point_msg.num_points)]
                point_msgs.pop(0)
        except IndexError:
            # We ran out of points
            pass

        if points is not None:
            for point in points:
                cv2.circle(image, point, radius=15, thickness=2, color=127)

        cv2.imshow(POINTS_CHANNEL, image)
        cv2.waitKey(frame_time)

    cv2.destroyAllWindows()


def check_prerequisites(log, channels):
    for channel in channels:
        msgs = [msg for msg in log if msg.channel == channel]
        assert len(msgs) > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a line follower",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--frame-time-msec",
                        help="Time each image is shown for (milleseconds)",
                        type=int,
                        default=75)
    parser.add_argument("-i", "--image-channel",
                        help="Channel to display images from",
                        default="IMAGE_RAW")
    parser.add_argument("logfile",
                        help="Logfile to display points for")
    args = parser.parse_args()

    main(lcm.EventLog(args.logfile, 'r'), args.image_channel,
                      args.frame_time_msec)
