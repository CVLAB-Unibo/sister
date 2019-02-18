import os
import glob
import argparse
import subprocess

marker_detector_script = '/home/daniele/work/workspace_cpp/aruco-3.0.11/build/utils/aruco_sister'
camera_file = '/home/daniele/work/workspace_cpp/aruco-3.0.11/data/camera.yml'
marker_size = 0.1
dataset_path = '/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/plate_multi_height/plate_rgb'


subfolders  = sorted([x[0] for x in os.walk(dataset_path)])


for sub in subfolders:
    images = glob.glob(os.path.join(sub, "*.png")) + glob.glob(os.path.join(sub, "*.jpg"))
    target = [x for x in images if 'center' in x]
    if len(target) > 0:
        for f in target:

            command = "{} {} -c {} -s {}".format(
                marker_detector_script,
                f,
                camera_file,
                marker_size
            )
            print(command)
            subprocess.call(command.split(' '))










