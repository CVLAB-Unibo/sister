#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import sys
import os
import glob
import cv2
import time
from sister import Utilities, Camera, SisterCamera, Reconstruction
from open3d import *
import argparse
import xmltodict


parser = argparse.ArgumentParser()
parser.add_argument("--camera_file", help="Camera parameters filename", type=str, required=True)
args = parser.parse_args()


# Camera
camera = SisterCamera(filename=args.camera_file)

reconstruction = Reconstruction(
    '/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/PrototypeModelGen2019/results/00004_disparity_multiview.png',
    None,  # '/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/PrototypeModelGen2019/source/00001_center.png',
    camera
)

reconstruction2 = Reconstruction(
    '/home/daniele/Downloads/image3.png',
    None,  # '/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/PrototypeModelGen2019/source/00001_center.png',
    camera
)

reconstruction2.cloud += np.array([0.5, 0, 0])
# Open3D Visualizatoin
draw_geometries([reconstruction.generatePCD(), reconstruction2.generatePCD()])

# Output file
write_point_cloud('/tmp/cloud.ply', pcd)
