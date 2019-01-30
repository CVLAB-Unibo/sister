#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import sys
import os
import glob
import cv2
import time
from sister import Utilities, Camera
from open3d import *
import argparse


def createPcd(cloud, color_image=None):
    pcd = PointCloud()
    pcd.points = Vector3dVector(cloud.reshape((-1, 3)))
    if color_image is not None:
        colors = color_image.astype(float).reshape((-1, 3)) / 255.
        pcd.colors = Vector3dVector(colors)

    return pcd


parser = argparse.ArgumentParser()
parser.add_argument("--camera_file", help="Camera parameters filename", type=str)
parser.add_argument("--depth_file", help="Depth filename", type=str)
parser.add_argument("--rgb_file", help="Rgb filename", type=str, default='')
parser.add_argument("--baseline", help="Stereo baseline", type=float, default=0.1)
parser.add_argument("--min_distance", help="Min clip distance", type=float, default=0.0)
parser.add_argument("--max_distance", help="Max clip distance", type=float, default=0.8)
args = parser.parse_args()


# Camera
camera = Camera(filename=args.camera_file)

# Input files
depth_file = args.depth_file
rgb_file = args.rgb_file

# Disparity&Depth
disparity = Utilities.loadRangeImage(depth_file)
depth = camera.getFx() * args.baseline / (disparity)
depth = np.clip(depth, args.min_distance, args.max_distance)

# RGB Image
rgb = None
if len(rgb_file) > 0:
    rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)

# Filtering
for i in range(0):
    depth = cv2.bilateralFilter(depth.astype(np.float32), 3, 0.5, 0)

kernel = np.ones((5, 5), np.float32)/1
depth = cv2.filter2D(depth.astype(np.float32), -1, kernel)


# Cloud generation
cloud = camera.depthMapToPointCloud(depth)

# Open3D Visualizatoin
pcd = createPcd(cloud, color_image=rgb)
draw_geometries([pcd])

# Output file
write_point_cloud('/tmp/cloud.ply', pcd)
