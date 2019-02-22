#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os
import glob
import cv2
from open3d.open3d.geometry import write_point_cloud, write_triangle_mesh
from open3d.open3d.visualization import draw_geometries
from sister.sister import Utilities, Camera
from sister.datasets import  CircularFrame
from open3d import *
import argparse

tags_scale_map = {
    "classical": 256.,
    "mcnn": 1.
}

parser = argparse.ArgumentParser("Test a Circular Frame")
parser.add_argument("--camera_file", help="Camera parameters filename", type=str, default='/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')
parser.add_argument("--path", help="Frame path", type=str, required=True)
parser.add_argument("--output_subfolder", help="Subfolder containing outputs", type=str, default='output')
parser.add_argument("--tag", help="Output tag", type=str, required=True)
parser.add_argument("--visualization_type", help="Visualziation Type", type=str, default="pcd")
parser.add_argument("--filter_level", help="Filtering Level", type=int, default=0)
parser.add_argument("--plane_level", help="Plane level", type=float, default=0.0)
parser.add_argument("--export", help="Export data", type=bool, default=False)
parser.add_argument("--debug", help="Debug mode", type=bool, default=False)
args = parser.parse_args()

# Camera
camera = Camera(filename=args.camera_file)

# Input files
path = args.path
tag = args.tag

# Frame
frame = CircularFrame(path)

# Output
output_folder = os.path.join(path, args.output_subfolder)
output_files = glob.glob(os.path.join(output_folder, '*'))
output_files = [x for x in output_files if tag in x]
if len(output_files)==1:
    output_file = output_files[0]
else:
    print("Error, Output not found! Or many results: ")
    for o in output_files:
        print("     - {}".format(o))
    sys.exit(0)


# Scaling factor computation
scale = 1.
for name, s in tags_scale_map.items():
    if name in output_file:
        scale = s

disparity = Utilities.loadRangeImage(output_file, scaling_factor=1./scale)

#Filtering
for i in range(args.filter_level):
    disparity = cv2.bilateralFilter(disparity.astype(np.float32), 5, 6, 6)


depth = camera.getFx() * frame.baseline() / disparity
camera_pose = frame.getPose('center')
max_z = np.abs(camera_pose[2,3] - args.plane_level)
depth = np.clip(depth, 0, max_z)

# RGB
rgb = frame.getImage('center')


print(np.min(depth),np.max(depth))
# Cloud generation
cloud = camera.depthMapToPointCloud(depth)




if args.visualization_type == 'pcd':
    pcd = Utilities.createPcd(cloud, color_image=rgb)
    pcd.transform(frame.getPose('center'))


    if args.export:
        output_folder = os.path.join(path, "clouds")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        write_point_cloud(os.path.join(output_folder,tag + ".ply"), pcd)
        write_point_cloud(os.path.join(output_folder, tag  + ".pcd"), pcd)

    if args.debug:
        draw_geometries([pcd])

elif args.visualization_type == 'mesh':
    mesh = Utilities.meshFromPointCloud(cloud, color_image=rgb)
    mesh.transform(frame.getPose('center'))
    if args.export:
        output_folder = os.path.join(path, "meshes")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        write_triangle_mesh(os.path.join(output_folder, tag + ".ply"), mesh)

    if args.debug:
        draw_geometries([mesh])