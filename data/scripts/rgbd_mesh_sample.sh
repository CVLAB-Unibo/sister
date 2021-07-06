#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml
RGB_FILE=/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/plate_multi_height/plate_rgbd/frame_00009.png
DEPTH_FILE=/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/plate_multi_height/plate_rgbd/frame_00009.png.depth.png
BASELINE=0.05
MIN_DISTANCE=0.2
MAX_DISTANCE=0.75
SCALING_FACTOR=1000
IS_DEPTH=true
VISUALIZATION_TYPE=mesh

python /home/daniele/work/workspace_python/sister/python/test_rf.py \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE \
--is_depth $IS_DEPTH