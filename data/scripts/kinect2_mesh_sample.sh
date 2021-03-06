#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/home/daniele/work/workspace_python/sister/data/cameras/kinect2.xml
RGB_FILE=/home/daniele/Downloads/snapshots_kinect/RGB3269.png
DEPTH_FILE=/home/daniele/Downloads/snapshots_kinect/DepthBig3269.tiff
BASELINE=0.005
MIN_DISTANCE=0.005
MAX_DISTANCE=0.8
SCALING_FACTOR=1
IS_DEPTH=1
VISUALIZATION_TYPE=mesh


python /home/daniele/work/workspace_python/sister/python/test_rf.py \
--is_depth 1 \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE