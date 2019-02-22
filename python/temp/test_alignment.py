from open3d import *

import numpy as np
import copy

from open3d.open3d.registration import registration_icp, TransformationEstimationPointToPoint


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

path1 = '/home/daniele/work/workspace_python/sister/dataset/clouds/component_0J/component_0J.ply'
#path2 = '/tmp/cloud.ply'
path2 = '/tmp/cloud.ply'

source = read_point_cloud(path1)
target = read_point_cloud(path2)

trans_init = np.loadtxt('/tmp/unity_support_test.0').reshape((4,4))

threshold = 0.0005
trans_init[:3,3] = np.array([0.,0.,0.602]).reshape((3,))


draw_registration_result(source, target, trans_init)
print("Initial alignment")
evaluation = evaluate_registration(source, target,
                                   threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = registration_icp(source, target, threshold, trans_init,
        TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
print("")
draw_registration_result(source, target, reg_p2p.transformation)

print("Apply point-to-plane ICP")
reg_p2l = registration_icp(source, target, threshold, trans_init,
        TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
print("")
draw_registration_result(source, target, reg_p2l.transformation)


#draw_registration_result(cloud1, cloud2)
#print("OK")