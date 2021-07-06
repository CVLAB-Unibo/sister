#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/Users/daniele/work/workspace_cpp/sister/data/cameras/ipad_video_camera.xml
RGB_FILE=/Users/daniele/Desktop/TempEyecan/tm_prototypes/ipad_test/outvideo/0005.jpg
DEPTH_FILE=/Users/daniele/Desktop/TempEyecan/tm_prototypes/ipad_test/outdepth/0005_depth.png
BASELINE=2.0
MIN_DISTANCE=0.01
MAX_DISTANCE=1.5
SCALING_FACTOR=1
IS_DEPTH=0
VISUALIZATION_TYPE=pcd


python /Users/daniele/work/workspace_cpp/sister/python/test_rf_monodepth.py \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE