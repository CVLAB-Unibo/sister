#!/bin/bash
template_path="CalibrationPlate.blend"
output_path="/tmp/sister_output"
T="0.0175869 -0.0922135 0.309748"
R="0.703367 0.0138492 0.00150031 -0.71069"
TF="0.0175869 -0.0922135 0.309748 0.703367 0.0138492 0.00150031 -0.71069"

### -b blender background option ###
blender $template_path -P blender_render.py -- -extrinsics $TF --sensor_width 33.984 -save_rgb -save_depth -output_path $output_path
python3 reconstruct_pcd.py --output_path $output_path --depth_path $output_path"/Depth/0000.exr" --save --visualize