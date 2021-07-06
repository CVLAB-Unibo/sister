import os
import glob
import argparse
import subprocess

template_path = '/home/daniele/work/workspace_python/sister/blender/CalibrationPlate.blend'
dataset_path = '/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/plate_multi_height/plate_rgb'
image_filename = '00000_center.png'
pose_filenmame = '00000_center.png_detection.txt'

subfolders  = sorted([x[0] for x in os.walk(dataset_path)])

for sub in subfolders:
    image = os.path.join(sub, image_filename)
    pose = os.path.join(sub, pose_filenmame)
    if 'baseline_4' not in sub:
        continue
    if os.path.exists(image) and os.path.exists(pose):

        outfolder = os.path.join(sub,"synth")
        TR = open(pose,"r").readline()


        syntch_command = 'blender {} -P blender_render.py -b -- -extrinsics {} -save_rgb -save_depth -output_path {}'.format(
            template_path,
            TR,
            outfolder
        )
        subprocess.call(syntch_command.split(' '))
        print(image,pose)

        generated_depth = os.path.join(outfolder, "Depth/0000.exr")
        generate_command = 'python3 reconstruct_pcd.py --output_path {} --depth_path {} --save'.format(
            outfolder,
            generated_depth
        )
        subprocess.call(generate_command.split(' '))



#blender $template_path -P blender_render.py -- -extrinsics $TF -save_rgb -save_depth -output_path $output_path
#python3 reconstruct_pcd.py --output_path $output_path --depth_path $output_path"/Depth/0000.exr" --save --visualize








