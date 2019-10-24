#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/Users/daniele/work/workspace_cpp/sister/data/cameras/usb_camera_hd.xml
RGB_FILE=/Users/daniele/Desktop/to_delete/TEMP/TestSkimap/00000_center.png
DEPTH_FILE=/Users/daniele/Desktop/to_delete/TEMP/TestSkimap/output/00000_classical_multiview.png
BASELINE=0.01
MIN_DISTANCE=0.01109500946883785
MAX_DISTANCE=0.2
SCALING_FACTOR=256
IS_DEPTH=0
VISUALIZATION_TYPE=pcd



python /Users/daniele/work/workspace_cpp/sister/python/test_rf.py \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE