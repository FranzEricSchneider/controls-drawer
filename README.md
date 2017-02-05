README

To run calibration
cd projects/controls-drawer
source env.sh

python software/drivers/logitech_hd1080p/camera_driver.py
python software/perception/basic_image_operations_filter.py
python software/drivers/cnc_table/position_driver_2d.py
cd results/calib_images/
python ../../software/controllers/exterior_camera_calibration.py
