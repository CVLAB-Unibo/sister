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

parser = argparse.ArgumentParser("Test Dataset")
parser.add_argument("--path", help="Dataset Path", type=str, required=True)
parser.add_argument("--output_folder", help="Output Folder", type=str, required=True)
parser.add_argument("--baseline_index", help="Baseline index [0,1,2,3,4]", type=int, required=True)
args = parser.parse_args()


camera = SisterCamera('/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')

side = 5
center_cross = 3
n = side * 4 + center_cross

images = sorted(glob.glob(os.path.join(args.path, "*.png")))

if len(images) % n == 0:
    print("Number of images is ok!")


if len(images) == n:
    print("One dataset found!")

    dataset = CircularDataset(args.path, camera=camera)
    print(len(dataset.images), dataset.expected_images)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    dataset.export(args.output_folder, baseline_index=args.baseline_index)

else:
    subsets = int(len(images) / n)
    print("Subsets found: ", subsets)

    for i in range(0, subsets):
        subimages = list(images[i*n:(i+1)*n])
        print("SUBIMAGES", subimages)
        dataset = CircularDataset(subimages, camera=camera)
        print("Subsets {} -> {} ".format(i, len(subimages)))

        output_folder = os.path.join(args.output_folder, "subset_{}_baseline_{}".format(i, args.baseline_index))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dataset.export(output_folder, baseline_index=args.baseline_index)

    pass


print(len(images))
sys.exit(0)


# cv2.imshow("center", dataset.getImage('center'))
# cv2.imshow("bottom_00", dataset.getImage('bottom_00'))
# cv2.imshow("top_00", dataset.getImage('top_00'))
# cv2.imshow("left_00", dataset.getImage('left_00'))
# cv2.imshow("right_00", dataset.getImage('right_00'))

# cv2.waitKey(0)
