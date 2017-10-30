###############################################################################
To run calibration
cd projects/controls-drawer
source env.sh

python software/drivers/cnc_table/position_driver_2d.py
python software/drivers/crenova_iscope_endoscope_2Mpix/camera_driver.py
python software/perception/basic_image_operations_filter.py
cd results/calib_images/
python ../../software/controllers/exterior_camera_calibration.py
# Illuminate the scene clearly with a phone (for now)
# Take a first 1-2 images and check that the thresholding is working well,
# change the threshold value as necessary

# Once you have images that you're happy with, kill all drivers
###############################################################################


###############################################################################
To run intrinsic camera calibration
cd projects/controls-drawer
source env.sh

# First, put all the pictures of the square calibration image for the Crenova
# 	 camera in crenova_iscope_endoscope_2Mpix/square_images/*.jpg
# Edit the file to change various things
subl software/drivers/general_camera_tests/test_calibration.py
python software/drivers/general_camera_tests/test_calibration.py
###############################################################################


###############################################################################
To move the table around by terminal

# Edit to get the commands you want
subl software/drivers/cnc_table/simple_stream.py
python software/drivers/cnc_table/simple_stream.py
###############################################################################


###############################################################################
To take pictures using the camera driver

# Run the camera driver with a flag that saves every image when triggered
python software/drivers/crenova_iscope_endoscope_2Mpix/camera_driver.py --save-images
python software/drivers/general_camera_tests/image_request.py
###############################################################################


###############################################################################
How to threshold images

# Look in basic_image_operations_filter.py to see the basic threshold code
# Turn this into a tool if you use it often
###############################################################################


###############################################################################
How to run the line follower
cd projects/controls-drawer
source env.sh

python software/drivers/cnc_table/position_driver_2d.py
python software/drivers/crenova_iscope_endoscope_2Mpix/camera_driver.py
python software/drivers/general_camera_tests/image_request.py --stream
python software/gui/display_images.py --image-channel IMAGE_TRACKING --include-points
[lcm-logger]
python software/controllers/line_follower.py
###############################################################################
