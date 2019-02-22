from open3d import *

import numpy as np
import copy
import argparse
import os

from open3d.open3d.geometry import read_point_cloud
from open3d.open3d.registration import registration_icp, TransformationEstimationPointToPoint, evaluate_registration
from open3d.open3d.visualization import draw_geometries


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

parser = argparse.ArgumentParser("Test Alignment")
parser.add_argument("--path", help="Frame path", type=str, required=True)
parser.add_argument("--cloud_model_path", help="Cloud of model to compare", type=str, required=True)
parser.add_argument("--tag", help="Output tag", type=str, required=True)
parser.add_argument("--debug", help="Debug", type=bool, default=False)
args = parser.parse_args()



path1 = args.cloud_model_path# '/home/daniele/work/workspace_python/sister/dataset/clouds/component_0J/component_0J.ply'
path2 = os.path.join(args.path, "clouds", args.tag+".ply")
pose_path = os.path.join(args.path, "..", 'object_pose.txt')

if not os.path.exists(pose_path):
    print("No Pose File for current object in: {}".format(pose_path))


source = read_point_cloud(path1)
target = read_point_cloud(path2)

trans_init = np.loadtxt(pose_path).reshape((4,4))

threshold = 0.0005
#trans_init[:3,3] = np.array([0.,0.,0.602]).reshape((3,))

if args.debug:
    draw_registration_result(source, target, trans_init)

evaluation = evaluate_registration(source, target, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = registration_icp(source, target, 0.001, trans_init, TransformationEstimationPointToPoint())
print(reg_p2p)

level_name = os.path.basename(args.path)
object_name = os.path.split(os.path.split(args.path)[0])[1]
print(object_name)
print(level_name)


print(reg_p2p.fitness)
print(reg_p2p.inlier_rmse)
print(len(reg_p2p.correspondence_set))

f = open('/tmp/results.txt','a')

f.write(level_name.split('_')[1])
f.write(' ' + level_name.split('_')[2])
f.write(' ' + str(reg_p2p.fitness))
f.write(' ' + str(reg_p2p.inlier_rmse))
f.write(' ' + str(len(reg_p2p.correspondence_set)))
f.write('\n')
f.close()

if args.debug:
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)
#
# print("Apply point-to-plane ICP")
# reg_p2l = registration_icp(source, target, 0.0005, trans_init,TransformationEstimationPointToPlane())
# print(reg_p2l)
# print("Transformation is:")
# print(reg_p2l.transformation)
# print("")
# if args.debug:
#     draw_registration_result(source, target, reg_p2l.transformation)


#draw_registration_result(cloud1, cloud2)
#print("OK")