#!/usr/bin/env bash

#!/usr/bin/env bash

CAMERA_FILE=/Users/daniele/work/workspace_cpp/sister/data/cameras/usb_camera.xml
RGB_FILE=/Users/daniele/Desktop/to_delete/ConveyorVideoDemo/labels_images/NTLoopForRecording_1_DCOLOR.png
DEPTH_FILE=/Users/daniele/Desktop/to_delete/ConveyorVideoDemo/labels_images/NTLoopForRecording_1_DDATA.png
BASELINE=0.01
MIN_DISTANCE=0
MAX_DISTANCE=255
SCALING_FACTOR=10
IS_DEPTH=0
VISUALIZATION_TYPE=mesh



python /Users/daniele/work/workspace_cpp/sister/python/test_rf_inverse_depth.py \
--camera_file $CAMERA_FILE \
--depth_file $DEPTH_FILE \
--rgb_file $RGB_FILE \
--baseline $BASELINE \
--min_distance $MIN_DISTANCE \
--max_distance $MAX_DISTANCE \
--scaling_factor $SCALING_FACTOR \
--visualization_type $VISUALIZATION_TYPE \
--is_depth $IS_DEPTH