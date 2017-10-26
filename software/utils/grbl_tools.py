import numpy as np
import time


def openSerial(serialConnection):
    serialConnection.write("\r\n\r\n")
    print("Wake up grbl")
    # Wait for grbl to initialize 
    time.sleep(2.0)
    print("Woken up!")
    # Flush startup text in serial input
    serialConnection.flushInput()


def setupBasicSystem(additional=[]):
    """
    Returns the basic setup that I use for the CNC table, plus any lines that
    you want to include
    """
    # GCODE list: http://www.cnccookbook.com/CCCNCGCodeRef.html
    basicLines = ["G21",        # Sets program coordinates to be in mm
                  "G91",        # Sets incremental programming of XYZ (now command 5 does +5 instead of global position 5)
                  "G94",        # Now feed mode is units/minute. Because of G21, this means mm/minute
                  "G01 F1500",  # Sets movement in a straight line, feedrate 1500 (mm/min as defined by G21/G94)
                  "$$"]         # Display the settings
    return basicLines + additional


def haltMessage():
    return ["X0 Y0"]


def makeCmd(xPos, yPos, speed):
    """
    Takes position and speed commands in m and m/s and returns in GRBL format
    """
    return "X{:.2f} Y{:.2f} F{}".format(mToMM(xPos), mToMM(yPos), mpsToMMPMin(speed))


def sendLines(serialConnection, lines, debug=False):
    if serialConnection.closed:
        raise RuntimeError("Told to send commands but serial port closed")

    # Stream g-code to grbl
    for line in lines:
        lineStrip = line.strip() # Strip all EOL characters for consistency

        # Send g-code block to grbl
        if debug:
            print("Sending: " + lineStrip)
        serialConnection.write(lineStrip + "\n")
        if debug:
            print("Sent!")

        # Wait for grbl response with carriage return
        if lineStrip == "$$":
            # Read the output ($$ is a request for grbl state, and returns more
            #   than a single line of output)
            grbl_out = []
            for idx in range(31):
                grbl_out.append(serialConnection.readline().strip())
                if debug:
                    print("grbl_out: {}".format(grbl_out[-1]))
        else:
            grbl_out = serialConnection.readline(30)
            if debug:
                print("grbl_out: {}".format(grbl_out))

    return grbl_out


def retreat(offerRetreat, serialConnection, position):
    position = np.array(position)
    if offerRetreat:
        # Wait here until grbl is finished to close serial port and file.
        shouldRetreat = raw_input("Program done. Do you want to retreat" +\
                                  " to the starting position? ([y]/n)")
        if shouldRetreat.lower() != 'n':
            retreatCorrect = raw_input("It looks like the head has moved"
                                       " {} m from the beginning, undo?"
                                       " ([y]/n)".format(position))
            if retreatCorrect.lower() != 'n':
                retreatSpeed = 0.025  # m/s (1500 mm/min)
                sleepTime = np.linalg.norm(position) / retreatSpeed + 1.5
                line = makeCmd(-position[0], -position[1], retreatSpeed)
                sendLines(serialConnection, [line], DEBUG_SENDLINES)
                print("Sleeping for {} seconds to move...".format(sleepTime))
                time.sleep(sleepTime)
                print("Continuing")
        print("Quitting")
    else:
        print("Instructed not to retreat, quitting")


def mToMM(value):
    ''' Converts values in m to mm. Works for m and m/s '''
    return value * 1e3


def mmToM(value):
    ''' Converts values in mm (int) to m (float). Works for mm and mm/s '''
    return value / 1.0e3


def mpsToMMPMin(value):
    ''' Converts values in m/s (float) to mm/min (int) '''
    return int(value * 1e3 * 60)
