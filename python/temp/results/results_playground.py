import numpy as np
import cv2
import os, glob

methods = [
    '00000_classical_horizontal',
    '00000_classical_multiview',
    '00000_classical_vertical',
    'CB_mccnn_raw',
    'CB_mccnn_refine',
    'CL_mccnn_raw',
    'CL_mccnn_refine',
    'CR_mccnn_raw',
    'CR_mccnn_refine',
    'CT_mccnn_raw',
    'CT_mccnn_refine',
    'FULL_mccnn_raw',
    'FULL_mccnn_refine',
    'HORIZONTAL_FULL_mccnn_raw',
    'HORIZONTAL_FULL_mccnn_refine',
    'HORIZONTAL_SUM_mccnn_raw',
    'HORIZONTAL_SUM_mccnn_refine',
    'SUM_mccnn_raw',
    'SUM_mccnn_refine',
    'VERTICAL_FULL_mccnn_raw',
    'VERTICAL_FULL_mccnn_refine',
    'VERTICAL_SUM_mccnn_raw',
    'VERTICAL_SUM_mccnn_refine',
]

methods = [
    'CB_mccnn_raw',
    'CL_mccnn_raw',
    'CR_mccnn_raw',
    'CT_mccnn_raw',
    'FULL_mccnn_raw',
    'HORIZONTAL_FULL_mccnn_raw',
    'VERTICAL_FULL_mccnn_raw',
    'SUM_mccnn_raw',
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

#results_path = '/home/daniele/Desktop/temp/SisterResults/zzio'
results_path = '/tmp/zzio2'

target_model = 'component_1B'
target_value = 2
level = 0
baseline = 2

bests = []
for method in methods:
    if 'raw' not in method and 'classical' not in method:
        continue

    results_name = target_model + "#" + method
    filename = os.path.join(results_path, results_name + ".txt")

    data = np.loadtxt(filename)
    data = data[level*6:level*6+6,:]

    rmse = np.min(data[:, target_value], axis=0)
    i_rmse = np.argmin(data[:, target_value], axis=0)

    bests.append((method, data[i_rmse], rmse))

bests = sorted(bests, key=lambda x: x[1][target_value])

for b in bests:
    print("{:30}  {:.7f} {:.2f} {:000d}".format(
        b[0],
        b[1][target_value],
        b[1][0],
        int(b[1][1]),
    ))
