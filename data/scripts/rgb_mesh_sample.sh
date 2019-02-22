#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml
RGB_FILE=/tmp/plinietto/subset_5_baseline_0/00000_center.png
DEPTH_FILE=/tmp/plinietto/subset_5_baseline_0/00000_NO_LR_disparity_horizontal.png
BASELINE=0.12
MIN_DISTANCE=0.005
MAX_DISTANCE=0.7
SCALING_FACTOR=256
IS_DEPTH=0
VISUALIZATION_TYPE=pcd



python /home/daniele/work/workspace_python/sister/python/test_rf.py \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE