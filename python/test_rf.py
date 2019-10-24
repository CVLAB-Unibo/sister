#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import sys
import os
import glob
import cv2
import time
from open3d.open3d.geometry import write_point_cloud, write_triangle_mesh
from open3d.open3d.visualization import draw_geometries
from sister.sister import Utilities, Camera
from open3d import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--camera_file", help="Camera parameters filename", type=str, required=True)
parser.add_argument("--depth_file", help="Depth filename", type=str, required=True)
parser.add_argument("--rgb_file", help="Rgb filename", type=str, default='')
parser.add_argument("--baseline", help="Stereo baseline", type=float, default=0.1)
parser.add_argument("--min_distance", help="Min clip distance", type=float, default=0.0)
parser.add_argument("--max_distance", help="Max clip distance", type=float, default=0.8)
parser.add_argument("--scaling_factor", help="Scaling factor s  -> will be applied (1/s)", type=float, default=256.)
parser.add_argument("--is_depth", help="Is input a depth?", type=bool, default=False)
parser.add_argument("--visualization_type", help="Visualziation Type", type=str, default="pcd")
args = parser.parse_args()


# Camera
camera = Camera(filename=args.camera_file)

# Input files
depth_file = args.depth_file
rgb_file = args.rgb_file

# Disparity&Depth
disparity = Utilities.loadRangeImage(depth_file, scaling_factor=1./args.scaling_factor)
#disparity = disparity[:1080, ::]

# DISPARITY SMOOTH
for i in range(0):
    disparity = cv2.bilateralFilter(disparity.astype(np.float32), 5, 6, 6)
for i in range(0):
    disparity = cv2.medianBlur(disparity.astype(np.float32), 5)


if args.is_depth:
    print("IT IS A DEPTH IMAGE!")
    depth = disparity
else:
    depth = camera.getFx() * args.baseline / (disparity)

print("MAX MIN", np.max(depth), np.min(depth))
depth = np.clip(depth, args.min_distance, args.max_distance)

# RGB Image
rgb = None
if len(rgb_file) > 0:
    rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)

# DEPTH SMOOTH
for i in range(10):
    depth = cv2.bilateralFilter(depth.astype(np.float32), 5, 0.01, 0)



# Cloud generation
cloud = camera.depthMapToPointCloud(depth)


# Open3D Visualizatoin


if args.visualization_type == 'pcd':

    pcd = Utilities.createPcd(cloud, color_image=rgb)
    draw_geometries([pcd])
    write_point_cloud('/tmp/cloud.ply', pcd)
elif args.visualization_type == 'mesh':
    mesh = Utilities.meshFromPointCloud(cloud, color_image=rgb)
    draw_geometries([mesh])
    write_triangle_mesh("/tmp/mesh.ply", mesh)
