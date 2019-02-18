#!/bin/bash
template_path="CalibrationPlate.blend"
output_path="./outputs"
T="0.017586899921298027 -0.09221349656581879 0.30974799394607544"
R="0.017355073243379593 0.021796815097332 -1.5809646844863892"

### -b blender background option ###
blender $template_path -P blender_render.py -- -extrinsics $T $R -save_rgb -save_depth -output_path $output_path
python3 reconstruct_pcd.py --output_path $output_path --depth_path $output_path"/Depth/0000.exr" --save --visualize