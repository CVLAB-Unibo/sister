from open3d import *

import numpy as np
import copy
import argparse
import os
import cv2
from sister.sister import *
from sister.datasets import *

from open3d.open3d.geometry import read_point_cloud
from open3d.open3d.registration import registration_icp, TransformationEstimationPointToPoint, evaluate_registration
from open3d.open3d.visualization import draw_geometries

camera = Camera(filename='/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')

path1 = '/home/daniele/Desktop/temp/SisterAlignmentScenes/arduino_0.exr'
# Frame
print("WOW")
frame_path ='/home/daniele/data/datasets/sister/v1/objects_full_scenes/arduino/level_5_120/'
output_path = os.path.join(frame_path,'output', '00000_classical_multiview'+'.png')
print(output_path)
frame = CircularFrame(frame_path)
scale = 256
disparity = Utilities.loadRangeImage(output_path, scaling_factor=1./scale)
depth2 = camera.getFx() * frame.baseline() / disparity




depth1 = cv2.imread(path1, cv2.IMREAD_ANYDEPTH)

mask = np.ones(depth1.shape, np.uint8) * 255
mask[depth1>100] = 0

depth1 = cv2.bitwise_and(depth1, depth1, mask=mask)
depth2 = cv2.bitwise_and(depth2, depth2, mask=mask)

min_gt = np.min(depth1)
max_gt = np.max(depth1)
print("GT RANGE {}/{}".format(min_gt, max_gt))
depth2 = np.clip(depth2,0, max_gt*1000)
print("SCENE RANGE: ",np.min(depth2),np.max(depth2))

diff = np.abs(depth2-depth1)
print("DIFF RANGE", np.min(diff), np.max(diff))


#print("COUNT:",np.count_nonzero(diff>th))
#diff[diff>th] = 1


normalized_diff = (diff - np.min(diff))/(np.max(diff)- np.min(diff))
# depth2_masked = depth2.copy()
# depth2_masked[mask<1] = 0
# diff = np.abs(depth1 - depth2)
# diff[mask<1] = 0
# print(np.min(diff))
# print(np.max(diff))

mse = np.square(depth2 - depth1).mean(axis=None)
rmse = np.sqrt(mse)
#mse = np.sum(np.abs(depth2 - depth1))
print("MEAN DIFF", mse, rmse)

normalized_depth1 = (depth1-np.min(depth1))/(np.max(depth1)-np.min(depth1))
normalized_depth2 = (depth2-np.min(depth2))/(np.max(depth2)-np.min(depth2))

cv2.imshow("depth1", normalized_depth1)
cv2.imshow("depth2", normalized_depth2)
cv2.imshow("diff",normalized_diff)

cv2.imshow("mask",mask)

cv2.moveWindow("depth1",20,20)
cv2.moveWindow("depth2",660,20)
cv2.moveWindow("diff",640+660,20)
cv2.waitKey(0)

