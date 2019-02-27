from open3d import *

import numpy as np
import copy
import argparse
import os
import cv2
import subprocess
from sister.datasets import *

command = 'python test_depths_match.py --dataset_path {} --model_name {} --gt_path {} --tag {} --filter_bounds 1 --output_file {}'

methods = [
    '00000_classical_horizontal',
    '00000_classical_multiview',
    '00000_classical_vertical',
    'CB_mccnn_raw',
    'CB_mccnn_refine',
    'CB_sgm',
    'CL_mccnn_raw',
    'CL_mccnn_refine',
    'CL_sgm',
    'CR_mccnn_raw',
    'CR_mccnn_refine',
    'CR_sgm',
    'CT_mccnn_raw',
    'CT_mccnn_refine',
    'CT_sgm',
    'FULL_mccnn_raw',
    'FULL_mccnn_refine',
    'FULL_sgm',
    'HORIZONTAL_FULL_mccnn_raw',
    'HORIZONTAL_FULL_mccnn_refine',
    'HORIZONTAL_FULL_sgm',
    'HORIZONTAL_SUM_mccnn_raw',
    'HORIZONTAL_SUM_mccnn_refine',
    'HORIZONTAL_SUM_sgm',
    'SUM_mccnn_raw',
    'SUM_mccnn_refine',
    'SUM_sgm',
    'VERTICAL_FULL_mccnn_raw',
    'VERTICAL_FULL_mccnn_refine',
    'VERTICAL_FULL_sgm',
    'VERTICAL_SUM_mccnn_raw',
    'VERTICAL_SUM_mccnn_refine',
    'VERTICAL_SUM_sgm'
]

models = [
    'arduino',
    'component_0J',
    'component_1B',
    'component_1G',
    'hexa_nut',
    'hexa_screw',
    'nodemcu',
    'washer',
]
dataset_path = '/home/daniele/data/datasets/sister/v1/objects_full_scenes'
gt_path = '/home/daniele/data/datasets/sister/v1/objects_full_scenes_gt'
output_path = '/tmp/final_results'

if not os.path.exists(output_path):
    os.makedirs(output_path)

for model in models:
    for method in methods:
        output_name = model + "#" + method
        output_filename = os.path.join(output_path, output_name+".txt")

        c = command.format(
            dataset_path,
            model,
            gt_path,
            method,
            output_filename
        )
        print(c)
        subprocess.call(c.split(' '))
