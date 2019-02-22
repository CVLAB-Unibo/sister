#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import sys
import os
import glob
import cv2
import time
from open3d import *
import argparse
import xmltodict
from sister.datasets import CircularDataset
from sister.sister import SisterCamera
import sister.transformations as trans

parser = argparse.ArgumentParser("Export Raw Dataset")
parser.add_argument("--path", help="Dataset Path", type=str, required=True)
parser.add_argument("--output_folder", help="Output Folder", type=str, required=True)
parser.add_argument("--subfolders_for_results", help="Creates a subfolders for results with this name if any", type=str, default="output")
parser.add_argument('--baselines', nargs='+', help='<Required> List of baselines contained in the raw dataset',
                    required=True)
parser.add_argument("--side", help="How many side? E.g. how many baselines changes?", type=int, default=1)
parser.add_argument("--center_cross", help="How many center images are stored?", type=int, default=3)
parser.add_argument("--prefix", help="Prefix name", type=str, default="level_")
args = parser.parse_args()

camera = SisterCamera('/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')

baselines = args.baselines

images = sorted(glob.glob(os.path.join(args.path, "*.png")))

# Count
total = len(images)
center_cross = args.center_cross
n =  4 + center_cross
bs = len(baselines)
hs = int(total / (n*bs))
print(baselines, bs, hs)

if len(images) % n == 0:
    print("Number of images is ok!")

if bs * hs == total:
    print("Number of baselines and heights is congruent!")

subsets = []
for i in range(0, len(images), n):
    subset = images[i:i+n]
    subsets.append(subset)


i = 0
for h in range(hs):
    for b in baselines:
        subset = subsets.pop(0)
        for i in range(n):
            name = CircularDataset.NAMES_FLOW[i]
            image_path = subset[i]
            pose_path = CircularDataset.getCorrespondingPose(image_path)
            print(name, image_path, pose_path)
            output_name = "{}_{}.{}".format(str(0).zfill(CircularDataset.GLOBAL_ZFILL), name, "png")
            pose_output_name = "{}_{}.{}".format(str(0).zfill(CircularDataset.GLOBAL_ZFILL), name, "txt")

            image = cv2.imread(image_path)
            pose = trans.matrix_from_pose(np.loadtxt(pose_path))

            output_folder = os.path.join(args.output_folder, "{}{}_{}".format(
                args.prefix,
                h,
                b
            ))


            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            subfolder = os.path.join(output_folder, args.subfolders_for_results)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            cv2.imwrite(os.path.join(output_folder, output_name), image)
            np.savetxt(os.path.join(output_folder, pose_output_name), pose)




