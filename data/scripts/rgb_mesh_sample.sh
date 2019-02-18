#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml
RGB_FILE=/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/plate_multi_height/plate_rgb/subset_1_baseline_4/00000_center.png
DEPTH_FILE=/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/Results/plate_multi_height/plate_rgb/subset_1_baseline_4/mccnn-raw.png
BASELINE=0.05
MIN_DISTANCE=0.01
MAX_DISTANCE=0.75
SCALING_FACTOR=1
IS_DEPTH=0
VISUALIZATION_TYPE=mesh

python /home/daniele/work/workspace_python/sister/python/test_rf.py \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE