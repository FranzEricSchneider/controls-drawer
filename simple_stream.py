#!/usr/bin/env python
"""
---------------------
The MIT License (MIT)

Copyright (c) 2012 Sungeun K. Jeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
---------------------
"""

from random import random
import serial
import time

# Open grbl serial port
s = serial.Serial('/dev/ttyUSB0', 115200)

# Wake up grbl
s.write("\r\n\r\n")
print("Wake up grbl")
time.sleep(2)   # Wait for grbl to initialize 
print("Woken up!")
s.flushInput()  # Flush startup text in serial input

cmd = []

lines = ["G21",
         "G91",
         "G94",
         # "G01 F1500",
         # "$$"]
         "$$",
         "G01 F1500",
         "X-10 Y-100"]

# scalar = 8
# for i in range(296 / scalar):
#     # speed = ((random() - 0.5) * 2) * 100 + 1500
#     speed = 1500
#     # lines += ["G01 F{0:.1f}".format(speed), "X{} Y0".format(scalar)]
#     lines += ["X{} Y0".format(scalar)]

#### for i in range(1000):
####     v = float(i) / 50
####     lines.append("X{0:.2f} Y{1:.2f}".format(1 - v, v))
####     v = 250 + (i + 1) * 0.1
####     lines.append("G01 F{}".format(v))
####     lines.append("X0.1 Y0")

# Stream g-code to grbl
for line in lines:
    l = line.strip() # Strip all EOL characters for consistency
    print "Sending: " + l,
    s.write(l + "\n") # Send g-code block to grbl
    print("Sent!")
    if l == "$$":
        for i in range(31):
            grbl_out = s.readline().strip()
            print("grbl_out: {}".format(grbl_out))
    else:
        grbl_out = s.readline(30) # Wait for grbl response with carriage return
        print("grbl_out: {}".format(grbl_out))
    # time.sleep(0.1)

# Wait here until grbl is finished to close serial port and file.
raw_input("  Press <Enter> to exit and disable grbl.") 

# Close file and serial port
s.close()