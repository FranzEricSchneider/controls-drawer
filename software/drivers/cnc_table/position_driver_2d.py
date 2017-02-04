#!/usr/bin/env python

import argparse
import numpy as np
import select
import serial
import time

import lcm

from utils import grbl_tools
from utils import lcm_msgs


DEBUG_HANDLE = False
DEBUG_SENDLINES = False
LOOP_TIMEOUT = 0.005  # Time in s
LOOP_SLEEP = 0.005    # Time in s


class positionDriver2D():
    def __init__(self, velocityLimit):
        self.velocityLimit = velocityLimit    # Velocities in m/s
        self.position = np.array([0.0, 0.0])  # Position in m
        self.running = False

        # Used to keep track of commands that need to be/were executed
        self.inMessages = []
        self.sentMessages = []

        self.lcmObj = lcm.LCM()
        self.inputChannel = "POSITION_COMMAND"
        self.outputChannel = "TOOL_STATE"
        self.subscription = self.lcmObj.subscribe(self.inputChannel,
                                                  self.handleCommands)

        # Open grbl serial port
        self.serialConnection = serial.Serial("/dev/ttyUSB0", 115200)

        # Wake up grbl (includes a sleep for several seconds)
        grbl_tools.openSerial(self.serialConnection)

        # Set up the table with the normal settings (mm/min, relative
        #   positioning, etc.)
        lines = grbl_tools.setupBasicSystem()
        grbl_tools.sendLines(self.serialConnection, lines, DEBUG_SENDLINES)

    def startRunning(self):
        '''
        Starts the event loop that handles incoming commands and takes position
        steps at the appropriate timesteps
        '''
        # Each time we call startRunning, set running to True for the while
        #   loop and False after that
        self.running = True
        timeLastSent = time.time()
        vectorLastSent = np.array([0.0, 0.0])
        # sentXVelocity = 0.0
        # sentYVelocity = 0.0
        firstLoopMsg = True

        while self.running:
            # Update estimation of state
            now = time.time()
            positionCmds = np.array([cmd.relative_position_m
                                     for cmd in self.sentMessages])
            # If we have sent commands W, X, Y, Z, this computes the tool
            #   position at the end of command Y. Theoretically, command Z is
            #   happening right now
            lastCmdPosition = np.sum(positionCmds[:-1], axis=0)
            distTravelled = (now - timeLastSent) * self.sentMessages[-1].command_v_mps
            self.position = lastCmdPosition + vectorLastSent * distTravelled

            # Send out state
            stateMsg = lcm_msgs.auto_instantiate(self.outputChannel)
            stateMsg.utime = lcm_msgs.utime_now()
            stateMsg.position_m[0] = self.position[0]
            stateMsg.position_m[1] = self.position[1]
            stateMsg.cycle_start = firstLoopMsg
            firstLoopMsg = False
            self.lcmObj.publish(self.outputChannel, stateMsg.encode())

            # Send out a new command if it is relevant
            try:
                now = time.time()
                distanceToTravel = np.linalg.norm(self.sentMessages[-1].relative_position_m)
                timeToTravel = distanceToTravel / self.sentMessages[-1].command_v_mps
                if now >= (timeLastSent + timeToTravel):
                    line = grbl_tools.makeCmd(self.inMessages[0].relative_position_m[0],
                                              self.inMessages[0].relative_position_m[1],
                                              self.inMessages[0].command_v_mps)
                    grbl_tools.sendLines(self.serialConnection, [line], DEBUG_SENDLINES)
                    self.sentMessages.append(self.inMessages.pop(0))
                    timeLastSent = now
                    vectorLastSent = np.array(self.sentMessages[-1].relative_position_m) /\
                                     np.linalg.norm(self.sentMessages[-1].relative_position_m)
            except IndexError:
                # An array hasn't been populated yet
                pass

            # Handle incoming requests if they've come in. If not, timeout and
            #   move on with your life
            lcm_msgs.lcmobj_handle_msg(self.lcmObj, LOOP_TIMEOUT)

            # Sleep in the loop
            time.sleep(LOOP_SLEEP)

        print("Out of the main loop! Cleaning up...")
        self.running = False
        self.cleanUp(offerRetreat=True)

    def handleCommands(self, channel, data):
        """
        If you get a command message, limit the velocity and save it to be sent
        to the table later
        """
        try:
            msg = lcm_msgs.auto_decode(channel, data)
            msg.command_v_mps = max(min(msg.command_v_mps, self.velocityLimit), 0.0)
            if DEBUG_HANDLE:
                print("Received msg:\n\tutime: {}\n\tvelocity: {}\n\tposition: "
                      "".format(msg.utime, msg.command_v_mps, msg.relative_position_m))
            self.inMessages.append(msg)
        except:
            self.cleanUp(offerRetreat=False)
            raise

    def cleanUp(self, offerRetreat=True):
        # Send a halt message
        self.running = False
        grbl_tools.sendLines(self.serialConnection, grbl_tools.haltMessage(), DEBUG_SENDLINES)

        # Retreat the head to the starting position if commanded
        grbl_tools.retreat(offerRetreat, self.serialConnection, self.position)

        # Close serial port to the CNC table
        self.serialConnection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts a 2D position driver",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 0.01 m/s is quivalent to 600 mm/min
    parser.add_argument("-v", "--velocity-limit",
                        help="Velocity (m/s) to limit vector velocity at",
                        type=float,
                        default=0.01)
    args = parser.parse_args()

    PD2D = positionDriver2D(args.velocity_limit)
    PD2D.startRunning()
