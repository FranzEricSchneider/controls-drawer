#!/usr/bin/env python

import argparse
import numpy as np
import select
import serial
import time

import lcm

from lcmtypes import lcm_velocity_t


DEBUG_HANDLE = False
DEBUG_RUNNINGCYCLE = True
DEBUG_SENDLINES = False
Y_V_LIMIT_MPS = 0.01  # In m/s (600 mm/min)


class VelocityDriver1D():
    def __init__(self, timeStep, xVelocity, xLimit):
        self.timeStep = timeStep                     # Time in s
        self.xVelocity = xVelocity                   # Velocities in m/s
        self.xStep = self.xVelocity * self.timeStep  # Position in m
        self.yVelocity = 0.0                         # Velocities in m/s
        self.xLimit = xLimit                         # Position in m
        self.xPosition = 0.0                         # Position in m
        self.yPosition = 0.0                         # Position in m
        self.loopTimeout = 0.005                     # Time in s
        self.running = False

        self.lcmObj = lcm.LCM()
        self.subscription = self.lcmObj.subscribe("V_COMMAND", self.handleCommands)

        # Open grbl serial port
        self.serialConnection = serial.Serial('/dev/ttyUSB0', 115200)

        # Wake up grbl
        self.serialConnection.write("\r\n\r\n")
        print("Wake up grbl")
        time.sleep(2)   # Wait for grbl to initialize 
        print("Woken up!")
        self.serialConnection.flushInput()  # Flush startup text in serial input

        # GCODE list: http://www.cnccookbook.com/CCCNCGCodeRef.html
        lines = ["G21",        # Sets program coordinates to be in mm
                 "G91",        # Sets incremental programming of XYZ (now command 5 does +5 instead of global position 5)
                 "G94",        # Now feed mode is units/minute. Because of G21, this means mm/minute
                 "G01 F1500",  # Sets movement in a straight line, feedrate 1500 (mm/min as defined by G21/G94)
                 "$$"]         # Display the settings
        self.sendLines(lines)

    def startRunning(self):
        '''
        Starts the event loop that handles incoming commands and takes position
        steps at the appropriate timesteps
        '''
        # Each time we call startRunning, set running to True for the while
        #   loop and False after that
        self.running = True
        timeLastSent = time.time()
        xPositionAtLastSend = self.xPosition
        yPositionAtLastSend = self.yPosition
        sentXVelocity = 0.0
        sentYVelocity = 0.0
        yStep = 0.0
        firstLoopMsg = True

        while self.xPosition <= self.xLimit and self.running:
            if time.time() >= (timeLastSent + self.timeStep):

                yStep = self.yVelocity * self.timeStep
                vVector = np.array([self.xVelocity, self.yVelocity])
                feedrateMPS = np.linalg.norm(vVector)
                line = "X{} Y{} F{}".format(mToMM(self.xStep),
                                            mToMM(yStep),
                                            mpsToMMPMin(feedrateMPS))
                
                # Do the bookkeeping - record what the current position is and
                #   what the last sent velocity was
                self.xPosition = xPositionAtLastSend + self.xStep
                self.yPosition = yPositionAtLastSend + yStep
                xPositionAtLastSend = self.xPosition
                yPositionAtLastSend = self.yPosition
                sentXVelocity = self.xVelocity
                sentYVelocity = self.yVelocity

                if DEBUG_RUNNINGCYCLE:
                    print('Sending combined step command')
                    print('\tConstant xStep = {}m, yStep = {}m'.format(self.xStep, yStep))
                    print('\txPosition: {}, yPosition: {}'.format(self.xPosition, self.yPosition))
                self.sendLines([line])
                timeLastSent = time.time()
            else:
                self.xPosition = xPositionAtLastSend + \
                                 (time.time() - timeLastSent) * sentXVelocity
                self.yPosition = yPositionAtLastSend + \
                                 (time.time() - timeLastSent) * sentYVelocity
                msg = lcm_velocity_t()
                msg.utime = long(time.time() * 1e6)
                msg.position_m[0] = self.xPosition
                msg.position_m[1] = self.yPosition
                msg.cycle_start = firstLoopMsg
                self.lcmObj.publish("HEAD_POSITION", msg.encode())
                firstLoopMsg = False

                # Wait for timeout to handle lcmObj, otherwise just pass
                rfds, wfds, efds = select.select([self.lcmObj.fileno()],
                                                 [], [], self.loopTimeout)
                if rfds:
                    self.lcmObj.handle()
            time.sleep(0.001)

        print("Out of the main loop! Cleaning up...")
        self.running = False
        self.cleanUp(giveRetreatOption=True)

    def cleanUp(self, giveRetreatOption=True):
        self.running = False
        self.sendLines(["X0 Y0"])

        if giveRetreatOption:
            # Wait here until grbl is finished to close serial port and file.
            shouldRetreat = raw_input("Program done. Do you want to retreat" +\
                                      " to the starting position? ([y]/n)")
            if shouldRetreat.lower() != 'n':
                retreatCorrect = raw_input("It looks like the head has moved" +\
                                           " ({}, {})m".format(self.xPosition,
                                                               self.yPosition) +\
                                           " from the beginning, undo? ([y]/n)")
                if retreatCorrect.lower() != 'n':
                    retreatSpeed = 0.025  # m/s (1500 mm/min)
                    sleepTime = np.ceil(
                        np.linalg.norm(
                            np.array([self.xPosition, self.yPosition])
                        ) / retreatSpeed
                    ) + 1.0
                    lines = ["X{} Y{} F{}".format(-mToMM(self.xPosition),
                                                  -mToMM(self.yPosition),
                                                  mpsToMMPMin(retreatSpeed))]
                    self.sendLines(lines)
                    print("Sleeping for {} seconds for move to complete...".format(sleepTime))
                    time.sleep(sleepTime)
                    print("Continuing")
            print("Quitting")
        else:
            print("Instructed not to retreat, quitting")

        # Close file and serial port
        self.serialConnection.close()

    def sendLines(self, lines):
        if self.serialConnection.closed:
            raise RuntimeError("Told to send commands but serial port closed")

        # Stream g-code to grbl
        for line in lines:
            lineStrip = line.strip() # Strip all EOL characters for consistency
            if 'x' in lineStrip.lower() and 'y' in lineStrip.lower():
                # For now, I'm referring to X and Y swapped from what the XPro
                #   thinks of it as. This corrects that if we are sending
                #   strings like "X10 Y20 F1500s"
                lineCorrected = swapXandY(lineStrip)
            else:
                lineCorrected = lineStrip

            if DEBUG_SENDLINES:
                print("Sending: " + lineCorrected)
            # Send g-code block to grbl
            self.serialConnection.write(lineCorrected + "\n")
            if DEBUG_SENDLINES:
                print("Sent!")

            if lineCorrected == "$$":
                for idx in range(31):
                    grbl_out = self.serialConnection.readline().strip()
                    if DEBUG_SENDLINES:
                        print("grbl_out: {}".format(grbl_out))
            else:
                # Wait for grbl response with carriage return
                grbl_out = self.serialConnection.readline(30)
                if DEBUG_SENDLINES:
                    print("grbl_out: {}".format(grbl_out))

    def handleCommands(self, channel, data):
        try:
            yCommand = lcm_velocity_t.decode(data).command_v_mps
            if abs(yCommand) <= Y_V_LIMIT_MPS:
                self.yVelocity = yCommand
            else:
                self.yVelocity = np.sign(yCommand) * Y_V_LIMIT_MPS
            if DEBUG_HANDLE:
                print("yVelocity: {}".format(self.yVelocity))
        except:
            self.cleanUp(giveRetreatOption=False)
            raise


def swapXandY(line):
    xIdx = line.index('X')
    yIdx = line.index('Y')
    line = line[:xIdx] + 'Y' + line[xIdx + 1:]
    line = line[:yIdx] + 'X' + line[yIdx + 1:]
    return line


def mToMM(value):
    ''' Converts values in m (float) to mm (int). Works for m and m/s '''
    return value * 1e3


def mmToM(value):
    ''' Converts values in mm (int) to m (float). Works for mm and mm/s '''
    return value / 1.0e3


def mpsToMMPMin(value):
    ''' Converts values in m/s (float) to mm/min (int) '''
    return value * 1e3 * 60


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts a 1D velocity driver",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--timestep-s",
                        help="Respond to position data every -t seconds",
                        type=float,
                        default=0.05)
    parser.add_argument("-v", "--x-velocity",
                        help="Velocity (mps) at which x axis should travel",
                        type=float,
                        default=0.01)  # Equivalent to 600 mm/min
    parser.add_argument("-x", "--x-limit",
                        help="Distance (m) at which the run should stop",
                        type=float,
                        default=0.1)
    args = parser.parse_args()

    VD1D = VelocityDriver1D(args.timestep_s, args.x_velocity, args.x_limit)
    VD1D.startRunning()
