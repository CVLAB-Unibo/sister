#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/home/daniele/work/workspace_python/sister/data/cameras/basler_camera.xml
RGB_FILE=/tmp/blts/0/ws/images/00000.jpg
DEPTH_FILE=/tmp/outframes/00000.png
BASELINE=1
MIN_DISTANCE=0.01
MAX_DISTANCE=0.5
SCALING_FACTOR=65536
IS_DEPTH=True
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
--is_depth $IS_DEPTH