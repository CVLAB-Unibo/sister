#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os
import glob
import cv2
from open3d.open3d.geometry import write_point_cloud, write_triangle_mesh
from open3d.open3d.visualization import draw_geometries
from sister.sister import Utilities, Camera
from sister.datasets import CircularFrame, ScaleManager
from open3d import *
import argparse

tags_scale_map = {
    "classical": 256.,
    "mcnn": 1.
}

parser = argparse.ArgumentParser("Test a Circular Frame")
parser.add_argument("--camera_file", help="Camera parameters filename", type=str,
                    default='/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')
parser.add_argument("--dataset_path", help="Dataset path", type=str, required=True)
parser.add_argument("--model_name", help="Model Name", type=str, required=True)
parser.add_argument("--level", help="Level", type=str, required=True)
parser.add_argument('--baselines', nargs='+', help='<Required> List of baselines contained in the raw dataset',
                    required=True)
parser.add_argument("--filter_level", help="Filtering Level", type=int, default=0)
parser.add_argument("--max_depth", help="Max distance", type=float, required=True)
parser.add_argument("--tag", help="Output tag", type=str, required=True)
parser.add_argument("--visualization_type", help="Visualziation Type", type=str, default="pcd")
parser.add_argument("--colors", help="Export data", type=bool, default=False)
parser.add_argument("--export", help="Export data", type=bool, default=False)
parser.add_argument("--debug", help="Debug mode", type=bool, default=False)

args = parser.parse_args()

# Camera
camera = Camera(filename=args.camera_file)

# Input files

tag = args.tag

depths = []
for b in args.baselines:

    # Frame
    subfolder_name = 'level_{}_{}'.format(args.level, str(b).zfill(3))
    framepath = os.path.join(args.dataset_path, args.model_name, subfolder_name)
    frame = CircularFrame(framepath)

    # Disparity
    fullpath = os.path.join(args.dataset_path, args.model_name, subfolder_name, 'output', args.tag + ".png")
    scale = ScaleManager.getScaleByName(args.tag)
    disparity = Utilities.loadRangeImage(fullpath, scaling_factor=1. / scale)

    # Filtering
    for i in range(args.filter_level):
        disparity = cv2.bilateralFilter(disparity.astype(np.float32), 5, 6, 6)

    # Depth
    print("Baseline:", frame.baseline())
    depth = camera.getFx() * frame.baseline() / disparity
    camera_pose = frame.getPose('center')
    depth = np.clip(depth, 0, args.max_depth)

    depths.append(depth)
    print(fullpath, os.path.exists(fullpath))

whole_depths = np.zeros(depths[0].shape).astype(float)
for dp in depths:
    whole_depths += dp.astype(float)

depth = whole_depths / float(len(depths))

# Cloud
cloud = camera.depthMapToPointCloud(depth)

if args.visualization_type == 'pcd':
    pcd = Utilities.createPcd(cloud, color_image=rgb if args.colors else None)
    pcd.transform(frame.getPose('center'))
    write_point_cloud('/tmp/{}_cloud.pcd'.format(args.model_name), pcd)
    if args.export:
        output_folder = os.path.join(path, "clouds")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        write_point_cloud(os.path.join(output_folder, tag + ".ply"), pcd)
        write_point_cloud(os.path.join(output_folder, tag + ".pcd"), pcd)

    if args.debug:
        draw_geometries([pcd])

elif args.visualization_type == 'mesh':
    mesh = Utilities.meshFromPointCloud(cloud, color_image=rgb if args.colors else None)
    mesh.transform(frame.getPose('center'))
    write_triangle_mesh('/tmp/{}.ply'.format(args.model_name), mesh)
    if args.export:
        output_folder = os.path.join(path, "meshes")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        write_triangle_mesh(os.path.join(output_folder, tag + ".ply"), mesh)

    if args.debug:
        draw_geometries([mesh])
