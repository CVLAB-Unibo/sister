from open3d import *
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Sister 3D Visualization Tool")
parser.add_argument('--depth', dest='visualize_depth', action='store_true', help='visualize depth')
parser.set_defaults(visualize_depth=True)
parser.add_argument("--depth_path", type=str, default="./outputs/Depth/0000.exr", help="input ply")
parser.add_argument("--threshold_depth", type=float, default=1, help="threshold depth")

parser.add_argument('--ply', dest='visualize_ply', action='store_true', help='visualize ply')
parser.set_defaults(visualize_ply=True)
parser.add_argument("--ply_path", type=str, default="./outputs/Pcd/pcd.ply", help="input ply")
args=parser.parse_args()

if args.visualize_depth:
    depth = cv2.imread(args.depth_path,cv2.IMREAD_UNCHANGED)[:,:,0]
    depth = (np.clip(depth,0,args.threshold_depth)/args.threshold_depth*255).astype(np.uint8)
    depth = cv2.applyColorMap(depth,2)
    cv2.imshow("Depth", depth)
    cv2.waitKey()

if args.visualize_ply:
    pcd=read_point_cloud(args.ply_path)
    pcd.paint_uniform_color([1, 0.706, 0])
    draw_geometries([pcd])
