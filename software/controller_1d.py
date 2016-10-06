#!/usr/bin/env python

import argparse
import numpy as np
import select
import serial
import signal
import time

import lcm

from lcmtypes import lcm_velocity_t


class Controller1D():
    def __init__(self, args):
        lcmObj = lcm.LCM()
        subscription = lcmObj.subscribe("HEAD_POSITION", self.positionHandler)
        self.plotX = []
        self.plotY = []

        try:
           while True:
                try:
                    lcmObj.handle()
                except:
                    # This is necessary because otherwise lcmObj goes through
                    #   the outer try except
                    raise
        except KeyboardInterrupt:
            import matplotlib.pyplot as plt
            plt.plot(self.plotX, self.plotY, '.-')
            plt.show()

    def positionHandler(self, channel, data):
        msg = lcm_velocity_t.decode(data)
        print("Received message on channel \"%s\"" % channel)
        print("\tutime   = %s" % str(msg.utime))
        print("\tcommand_v_mps = %s" % str(msg.command_v_mps))
        print("\tposition    = %s" % str(msg.position_m))
        print("")
        self.plotX.append(msg.utime)
        self.plotY.append(msg.position_m[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starts a 1D controller',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-a', '--analyze-scale-data',
    #                     help='Runs scale analysis. May not want to run scale analysis if the items are inconsistent weights',
    #                     action='store_true')
    args = parser.parse_args()
    C1D = Controller1D(args)
