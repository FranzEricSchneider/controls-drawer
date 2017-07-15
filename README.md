README

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
To move the table around by terminal

subl software/drivers/cnc_table/simple_stream.py  # Edit to get the commands you want
python software/drivers/cnc_table/simple_stream.py
###############################################################################
