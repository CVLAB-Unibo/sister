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

parser = argparse.ArgumentParser("Test Dataset")
parser.add_argument("--path", help="Dataset Path", type=str, required=True)
parser.add_argument("--output_folder", help="Output Folder", type=str, required=True)
parser.add_argument("--baseline_index", help="Baseline index [0,1,2,3,4]", type=int, required=True)
args = parser.parse_args()


dataset = CircularDataset(args.path)
print(len(dataset.images), dataset.expected_images)

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

dataset.export(args.output_folder, baseline_index=args.baseline_index)
# cv2.imshow("center", dataset.getImage('center'))
# cv2.imshow("bottom_00", dataset.getImage('bottom_00'))
# cv2.imshow("top_00", dataset.getImage('top_00'))
# cv2.imshow("left_00", dataset.getImage('left_00'))
# cv2.imshow("right_00", dataset.getImage('right_00'))

# cv2.waitKey(0)
