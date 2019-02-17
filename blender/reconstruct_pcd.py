import argparse
import os
import numpy as np
from open3d import *
import cv2
import time
import datetime

from utils_transform import euler_matrix

parser = argparse.ArgumentParser(description="Sister Ply Reconstruction Tool")
parser.add_argument("--sensor_width", dest='sensor_width', type=float, default=32,  help="sensor width in mm")

parser.add_argument("--focal_x", dest='focal_x', type=float, default=528, help="camera focal lenght in mm or pixel")
parser.add_argument("--focal_y", dest='focal_y', type=float, default=528, help="camera focal lenght in mm or pixel")

parser.add_argument('--focal_in_mm', dest='focal_in_mm', action='store_true', help='consider focal lenght in mm, default in pixel')
parser.set_defaults(focal_in_mm=False)

parser.add_argument("--c_x", dest='c_x', type=float, default=320 , help="c_x") #334.548055
parser.add_argument("--c_y", dest='c_y', type=float, default=240 , help="c_y") #237.481637

parser.add_argument('--extrinsics', dest='extrinsics', type=float, nargs='+', default=[0.017586899921298027, -0.09221349656581879, 0.30974799394607544, 0.017355073243379593, 0.021796815097332, -1.5809646844863892], help='extrinscs in format Tx Ty Tz Rx Ry Rz, meter and radians')

parser.add_argument("--depth_path", type=str, default="./outputs/Depth/0000.exr", help="input depht map")
parser.add_argument("--output_path", type=str, default="./outputs", help="where output files will be stored")

parser.add_argument('--save_ply', dest='save_ply', action='store_true', help='save ply view')
parser.set_defaults(save_ply=False)

parser.add_argument('--visualize_ply', dest='visualize_ply', action='store_true', help='visualize ply')
parser.set_defaults(visualize_ply=False)
args=parser.parse_args()

def reconstruct_pcd(depth, focal_x, focal_y, c_x, c_y, th_depth=100):
    pcd=[]    
    # K = np.asarray([[focal_x, 0 ,c_x],
    #      [0 , focal_y ,c_y],
    #      [0, 0 ,1]])
    # print(K)
    # K_inv = np.linalg.inv(K)
    for i in range(depth.shape[1]):
        for j in range(depth.shape[0]):
        # i=75
        # j=383
        if depth[j,i] < th_depth:
            # p=np.dot(K_inv, np.asarray([i,j,1]))*depth[j,i]
            # print(p)
            px = (i - c_x) * depth[j,i] / focal_x
            py = (j- c_y) * depth[j,i] / focal_y
            pz = depth[j,i]
            p=[px,py,pz]
        pcd.append(p)
    print("Total number of points:", len(pcd))
    return np.asarray(pcd)

start = time.time()
depth = cv2.imread(args.depth_path,cv2.IMREAD_UNCHANGED)[:,:,0]

width = depth.shape[0]
height = depth.shape[1]

translation_camera = args.extrinsics[:3]
rotation_camera = args.extrinsics[3:]

### suppose same same ratio width/height bewteen sensor and resolution
if args.focal_in_mm:
    focalx_in_pixel = args.focal_x * width / args.sensor_width
    focaly_in_pixel = args.focal_y * width / args.sensor_width
else:
    focalx_in_pixel = args.focal_x
    focaly_in_pixel = args.focal_y

xyz = reconstruct_pcd(depth, focalx_in_pixel, focaly_in_pixel, args.c_x, args.c_y)

Tr=euler_matrix(rotation_camera[0],rotation_camera[1],rotation_camera[2],'sxyz')
Tr[:,3]= np.asarray([translation_camera[0],translation_camera[1],translation_camera[2],1.0])
Tr_inv = np.linalg.inv(Tr)

pcd = PointCloud()
pcd.points = Vector3dVector(xyz)
pcd.transform(Tr_inv)

print("Time: ", datetime.timedelta(seconds=time.time() - start))

if args.save_ply:
    if not os.path.exists(os.path.join(args.output_path,"Pcd")):
        os.makedirs(os.path.join(args.output_path,"Pcd"))
    write_point_cloud(os.path.join(args.output_path,"Pcd","pcd.ply"), pcd)

if args.visualize_ply:
    pcd.paint_uniform_color([1, 0.706, 0])
    draw_geometries([pcd])