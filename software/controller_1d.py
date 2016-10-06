#!/usr/bin/env python

import argparse
import numpy as np
import select
import serial
import signal
import time

import lcm

from lcmtypes import lcm_velocity_t


DEBUG_HANDLE = False

CONSTANT_FN_VALUE_M = 0.01

SINE_FREQ_RPS = np.pi / 2  # /2pi = cycles per second (1/4 cycle per second)
SINE_AMPLITUDE_M = 0.02
SINE_OFFSET_M = 0.0
SINE_ANGULAR_OFFSET_R = np.pi

P_CONSTANT = 0.5  # 1cm offset results in reaction of 0.005 m/s, 300 mm/min


class Controller1D():
    def __init__(self, referencePointFn, controllerFn, timestep):
        self.referencePointFn = referencePointFn
        self.controllerFn = controllerFn
        self.timestep = timestep

        self.lcmObj = lcm.LCM()
        self.subscription = self.lcmObj.subscribe("HEAD_POSITION",
                                                  self.positionHandler)
        self.cycleStartUtime = long(0)
        self.timeLastSent = time.time()
        self.plotX = []
        self.plotY = []

        try:
           while True:
                try:
                    self.lcmObj.handle()
                except:
                    # This is necessary because otherwise lcmObj goes through
                    #   the outer try except
                    raise
        except KeyboardInterrupt:
            import matplotlib.pyplot as plt
            plt.plot(self.plotX, self.plotY, '.-')
            plt.show()

    def positionHandler(self, channel, data):
        rcvMsg = lcm_velocity_t.decode(data)
        if rcvMsg.cycle_start:
            self.cycleStartUtime = rcvMsg.utime

        if time.time() >= self.timeLastSent + self.timestep:
            referencePoint = self.referencePointFn(rcvMsg.utime - self.cycleStartUtime)
            vCommand = self.controllerFn(referencePoint - rcvMsg.position_m[1])
            sendMsg = lcm_velocity_t()
            sendMsg.utime = long(time.time() * 1e6)
            sendMsg.command_v_mps = vCommand
            self.lcmObj.publish("V_COMMAND", sendMsg.encode())
            self.timeLastSent = time.time()

        if DEBUG_HANDLE:
            print("Received message on channel \"%s\"" % channel)
            print("\tutime         = %s" % str(rcvMsg.utime))
            print("\tcommand_v_mps = %s" % str(rcvMsg.command_v_mps))
            print("\tposition      = %s" % str(rcvMsg.position_m))
            print("\tcycle_start   = %s" % str(rcvMsg.cycle_start))
        self.plotX.append(usToS(rcvMsg.utime))
        self.plotY.append(rcvMsg.position_m[1])


def constantFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    '''
    return CONSTANT_FN_VALUE_M


def sineFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    '''
    t = usToS(dUT)
    return (np.sin((SINE_FREQ_RPS * t) + SINE_ANGULAR_OFFSET_R) * SINE_AMPLITUDE_M) + SINE_OFFSET_M


def pCtrlr(eM):
    '''
    input (eM): error in meters
    returns: velocity setpoint of reacting system (m/s)
    '''
    return P_CONSTANT * eM


def usToS(utime):
    ''' Converts values in microseconds to values in seconds '''
    return (utime / 1.0e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts a 1D controller",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--controller-function",
                        help="Choose which function to use as a controller",
                        action="store",
                        default="proportional")
    parser.add_argument("-s", "--reference-point-function",
                        help="Choose which function to use as a referencePoint",
                        action="store",
                        default="constant")
    parser.add_argument("-t", "--timestep-s",
                        help="Respond to position data every -t seconds",
                        type=float,
                        default=0.01)
    args = parser.parse_args()

    if args.reference_point_function == "constant":
        referencePointFn = constantFn
    elif args.reference_point_function == "sine":
        referencePointFn = sineFn
    else:
        raise ValueError("Didn't recognize referencePointFn {}".format(args.reference_point_function))

    if args.controller_function == "proportional":
        controllerFn = pCtrlr
    else:
        raise ValueError("Didn't recognize controllerFn".format(args.controller_function))

    C1D = Controller1D(referencePointFn, controllerFn, args.timestep_s)
