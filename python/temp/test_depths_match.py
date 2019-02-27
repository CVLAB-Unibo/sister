from open3d import *

import numpy as np
import copy
import argparse
import os
import cv2
from sister.sister import *
from sister.datasets import *


def normalized(d):
    return (d - np.min(d)) / (np.max(d) - np.min(d))


parser = argparse.ArgumentParser("Test Alignment")
parser.add_argument("--camera_file", help="Camera parameters filename", type=str,
                    default='/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')
parser.add_argument("--dataset_path", help="Dataset path", type=str, required=True)
parser.add_argument("--model_name", help="Model path", type=str, required=True)
parser.add_argument("--gt_path", help="Ground truth path", type=str, required=True)
parser.add_argument("--tag", help="Output tag", type=str, required=True)
parser.add_argument("--debug", help="Debug", type=bool, default=False)
parser.add_argument("--plane_level", help="Plane level", type=float, default=0.0432)
parser.add_argument("--filter_bounds", help="Fitler range bounds", type=bool, default=False)
parser.add_argument("--output_file", help="Output file", type=str, default="")

args = parser.parse_args()

camera = Camera(filename=args.camera_file)

model_path = os.path.join(args.dataset_path, args.model_name)
frames_paths = [x for x in sorted(glob.glob(os.path.join(model_path, "*"))) if os.path.isdir(x)]

results = []
for f in frames_paths:
    # BUILDS PATHS
    basename = os.path.basename(f)

    level = basename.split('_')[1]
    baseline = basename.split('_')[2]
    gt_name = args.model_name + '_' + level + '.exr'
    gt_path = os.path.join(args.gt_path, gt_name) if len(args.gt_path) > 0 else None
    print("#" * 10)
    print(basename, level, baseline, gt_path)

    # LOAD CORRESPONDING OUTPUT FILE
    output_path = [x for x in glob.glob(os.path.join(f, 'output', '*.png')) if args.tag == os.path.splitext(os.path.basename(x))[0]]
    if len(output_path) != 1:
        print("ERROR! {}, not found".format(output_path))
        continue
    output_path = output_path[0]
    print(output_path)

    # GT DEPTH
    depth_gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)

    # FRAME
    frame = CircularFrame(f)
    scale = ScaleManager.getScaleByName(args.tag)
    disparity = Utilities.loadRangeImage(output_path, scaling_factor=1. / scale)
    depth = camera.getFx() * frame.baseline() / disparity

    # MASK
    mask = np.ones(depth_gt.shape, np.uint8) * 255
    mask[depth_gt > 100] = 0


    # MASKING
    depth_gt = cv2.bitwise_and(depth_gt, depth_gt, mask=mask)
    depth = cv2.bitwise_and(depth, depth, mask=mask)
    #depth_gt[depth_gt > 100] = np.fabs(frame.getPose("center")[3,3]) - args.plane_level





    # FILTER BOUNDS
    if args.filter_bounds:
        print("Filtering bounds...")
        min_gt = np.min(depth_gt)
        max_gt = np.max(depth_gt)
        depth = np.clip(depth, 0, max_gt*2)#max_gt*10)
        print("SCENE RANGE: ", np.min(depth), np.max(depth))

    # DIFF
    diff = np.abs(depth_gt - depth)
    print("DIFF RANGE", np.min(diff), np.max(diff))
    normalized_diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))

    # METRICS
    mae = np.abs(depth - depth_gt).mean(axis=None)
    mse = np.square(depth - depth_gt).mean(axis=None)
    rmse = np.sqrt(mse)

    # ACCURACTY
    ratio_0 = np.divide(depth, depth_gt)
    ratio_1 = np.divide(depth, depth_gt)
    ratio = np.maximum(ratio_0, ratio_1)

    acc_th = 1.01
    inliers = np.count_nonzero(ratio <= acc_th)
    accuracy = inliers / float(np.count_nonzero(mask))
    results.append((level, baseline, mae, mse, rmse, inliers, accuracy))

    if args.debug:
        print("Mae: ", mae, " Mse:", mse, " RMSE:", rmse, "TOTAL",np.count_nonzero(mask), "INLIERS:", inliers, " ACC:", accuracy)
        #cv2.imshow("depth", normalized(depth))

        print("MIN MAX DEPTH", np.min(depth), np.max(depth))
        print("MIN MAX GT", np.min(depth_gt), np.max(depth_gt))
        cv2.imshow("depth", normalized(depth))
        cv2.imshow("depth_gt", normalized(depth_gt))
        cv2.imshow("diff", normalized_diff)
        cv2.imshow("mask", mask)

        cv2.moveWindow("depth", 20, 20)
        cv2.moveWindow("depth_gt", 660, 20)
        cv2.moveWindow("diff", 640 + 660, 20)
        cv2.moveWindow("mask", 20, 20+480)
        cv2.waitKey(0)



if len(args.output_file)>0:

    f = open(args.output_file, 'w')
    f.write("# LEVEL BASELINE MAE MSE RMSE INLIERS ACC_1.05\n")
    for r in results:
        f.write(' '.join(map(str, r)))
        f.write('\n')
    f.close()

