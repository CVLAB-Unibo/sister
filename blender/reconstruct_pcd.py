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

parser.add_argument("--focal_x", dest='focal_x', type=float, default=528.580921, help="camera focal lenght in mm or pixel")
parser.add_argument("--focal_y", dest='focal_y', type=float, default=528.000913, help="camera focal lenght in mm or pixel")

parser.add_argument('--focal_in_mm', dest='focal_in_mm', action='store_true', help='consider focal lenght in mm, default in pixel')
parser.set_defaults(focal_in_mm=False)

parser.add_argument("--c_x", dest='c_x', type=float, default=334.548055 , help="c_x") #334.548055
parser.add_argument("--c_y", dest='c_y', type=float, default=237.481637 , help="c_y") #237.481637

parser.add_argument('--extrinsics', dest='extrinsics', type=float, nargs='+', default=[0.017586899921298027, -0.09221349656581879, 0.30974799394607544, 0.017355073243379593, 0.021796815097332, -1.5809646844863892], help='extrinscs in format Tx Ty Tz Rx Ry Rz, meter and radians')

parser.add_argument("--crop_h", dest='crop_h', type=int, default=-1,  help="crop depth")
parser.add_argument("--crop_w", dest='crop_w', type=int, default=-1,  help="crop depth")
parser.add_argument("--offset_h", dest='offset_h', type=int, default=0,  help="crop depth")
parser.add_argument("--offset_w", dest='offset_w', type=int, default=0,  help="crop depth")

parser.add_argument("--depth_path", type=str, default="./outputs/Depth/0000.exr", help="input depht map")
parser.add_argument("--tgb_path", type=str, default="./outputs/Depth/0000.exr", help="input depht map")

parser.add_argument("--output_path", type=str, default="./outputs", help="where output files will be stored")

parser.add_argument('--save_ply', dest='save_ply', action='store_true', help='save ply view')
parser.set_defaults(save_ply=False)

parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', help='save mesh')
parser.set_defaults(save_mesh=False)

parser.add_argument('--visualize_mesh', dest='visualize_mesh', action='store_true', help='save ply view')
parser.set_defaults(visualize_mesh=False)

parser.add_argument('--worlds_coordinates', dest='worlds_coordinates', action='store_true', help='Transforms pcd in worlds coordinates')

parser.add_argument('--visualize_ply', dest='visualize_ply', action='store_true', help='visualize ply')
parser.set_defaults(visualize_ply=False)
args=parser.parse_args()

def createPcd(cloud, color_image=None):
    pcd = PointCloud()
    pcd.points = Vector3dVector(cloud.reshape((-1, 3)))
    if color_image is not None:
        colors = color_image.astype(float).reshape((-1, 3)) / 255.
        pcd.colors = Vector3dVector(colors)
    return pcd

def meshFromPointCloud(cloud, color_image=None, max_perimeter_threshold=0.5):
    pcd = createPcd(cloud)

    mesh = TriangleMesh()

    mesh.vertices = pcd.points
    print(mesh.vertices)
    print(mesh.triangles)

    w = cloud.shape[1]
    h = cloud.shape[0]
    size = w*h
    size_faces = (w-1)*(h-1)*2
    points = np.zeros((size, 3), float)
    colors = np.zeros((size, 3), float)
    triangles = np.zeros((size_faces, 3), np.int32)
    color_map = np.ones((h, w, 3))*100 if color_image is None else color_image

    color_map = color_map / 255.

    points_index = 0
    triangle_index = 0
    for r in range(0, cloud.shape[0], 1):
        for c in range(0, cloud.shape[1], 1):
            p0 = cloud[r, c, :]

            if r < cloud.shape[0]-1 and c < cloud.shape[1]-1:
                p1 = cloud[r, c+1, :]
                p2 = cloud[r+1, c, :]
                p3 = cloud[r+1, c+1, :]

                i0 = r * w + c
                i1 = r * w + c + 1
                i2 = (r+1)*w + c
                i3 = (r+1)*w + c + 1

                tmp1= np.linalg.norm(p1-p2)
                tmp2=np.linalg.norm(p2-p3) 
                tmp3=np.linalg.norm(p3-p1)
                perimeter = tmp1 + tmp2 + tmp3

                if p1[2] < 0.01 or p2[2] < 0.01 or p3[2] < 0.01 or perimeter > max_perimeter_threshold:
                    triangles[triangle_index, :] = np.array([0, 0, 0])
                    triangles[triangle_index+1, :] = np.array([0, 0, 0])
                else:
                    triangles[triangle_index, :] = np.array([i0, i2, i1])
                    triangles[triangle_index+1, :] = np.array([i1, i2, i3])
                # triangles.append([i0, i1, i2])
                triangle_index += 2

                # triangles.append([i1, i3, i2])
            points[points_index, :] = p0
            colors[points_index, :] = color_map[r, c, :]
            points_index += 1

    mesh.triangles = open3d.Vector3iVector(np.array(triangles))
    mesh.vertices = open3d.Vector3dVector(np.array(points))
    mesh.vertex_colors = open3d.Vector3dVector(np.array(colors))
    # mesh.triangle_normals = open3d.Vector3dVector(np.array(normals))
    mesh.compute_vertex_normals()

    return mesh

def reconstruct_pcd(depth, focal_x, focal_y, c_x, c_y, th_depth=1, crop_w=-1, crop_h=-1, offset_w=0, offset_h=0):
    pcd=[]
    pcd_all=np.zeros([crop_h,crop_w,3])

    if crop_h < 0 or crop_w < 0:
        crop_w = depth.shape[1]
        crop_h = depth.shape[0]
    
    mean_depth=np.mean(depth[depth<th_depth])
    # K = np.asarray([[focal_x, 0 ,c_x],
    #      [0 , focal_y ,c_y],
    #      [0, 0 ,1]])
    # print(K)
    # K_inv = np.linalg.inv(K)
    # p=np.dot(K_inv, np.asarray([i,j,1]))*depth[j,i]
    # print(p)

    for i in range(offset_w,offset_w+crop_w):
        for j in range(offset_h,offset_h+crop_h):
        # i=75
        # j=383
            if depth[j,i] < th_depth:
                px = (i - c_x) * depth[j,i] / focal_x
                py = (j- c_y) * depth[j,i] / focal_y
                pz = depth[j,i]
                p=[px,py,pz]
                pcd.append(p)
                pcd_all[j-offset_h,i-offset_w]=np.asarray(p)
            else:
                pcd_all[j-offset_h,i-offset_w]=np.asarray([0,0,mean_depth])

    print("Total number of points:", len(pcd))
    return np.asarray(pcd),pcd_all

start = time.time()
depth = cv2.imread(args.depth_path,cv2.IMREAD_UNCHANGED)

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

xyz, xyz_all = reconstruct_pcd(depth, focalx_in_pixel, focaly_in_pixel, args.c_x, args.c_y, crop_w=args.crop_w, crop_h=args.crop_h, offset_h=args.offset_h, offset_w=args.offset_w )

if args.visualize_mesh:
    mesh=meshFromPointCloud(xyz_all)
    draw_geometries([mesh])

if args.save_mesh:    
    if not os.path.exists(os.path.join(args.output_path,"Mesh")):
        os.makedirs(os.path.join(args.output_path,"Mesh"))
    write_triangle_mesh(os.path.join(args.output_path,"Mesh","mesh.ply"), mesh)

pcd = PointCloud()
pcd.points = Vector3dVector(xyz)

if args.worlds_coordinates:
    Tr=euler_matrix(rotation_camera[0],rotation_camera[1],rotation_camera[2],'sxyz')
    Tr[:,3]= np.asarray([translation_camera[0],translation_camera[1],translation_camera[2],1.0])
    Tr_inv = np.linalg.inv(Tr)
    pcd.transform(Tr_inv)

print("Time: ", datetime.timedelta(seconds=time.time() - start))

if args.save_ply:
    if not os.path.exists(os.path.join(args.output_path,"Pcd")):
        os.makedirs(os.path.join(args.output_path,"Pcd"))
    write_point_cloud(os.path.join(args.output_path,"Pcd","pcd.ply"), pcd)

if args.visualize_ply:
    pcd.paint_uniform_color([1, 0.706, 0])
    draw_geometries([pcd])