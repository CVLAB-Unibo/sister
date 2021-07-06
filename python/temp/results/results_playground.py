import numpy as np
import cv2
import os, glob
from sister.datasets import BunchOfResults

#
# methods = [
#     '00000_classical_horizontal',
#     '00000_classical_multiview',
#     '00000_classical_vertical',
#     'CB_mccnn_raw',
#     'CB_mccnn_refine',
#     'CB_sgm',
#     'CL_mccnn_raw',
#     'CL_mccnn_refine',
#     'CL_sgm',
#     'CR_mccnn_raw',
#     'CR_mccnn_refine',
#     'CR_sgm',
#     'CT_mccnn_raw',
#     'CT_mccnn_refine',
#     'CT_sgm',
#     'FULL_mccnn_raw',
#     'FULL_mccnn_refine',
#     'FULL_sgm',
#     'HORIZONTAL_FULL_mccnn_raw',
#     'HORIZONTAL_FULL_mccnn_refine',
#     'HORIZONTAL_FULL_sgm',
#     'HORIZONTAL_SUM_mccnn_raw',
#     'HORIZONTAL_SUM_mccnn_refine',
#     'HORIZONTAL_SUM_sgm',
#     'SUM_mccnn_raw',
#     'SUM_mccnn_refine',
#     'SUM_sgm',
#     'VERTICAL_FULL_mccnn_raw',
#     'VERTICAL_FULL_mccnn_refine',
#     'VERTICAL_FULL_sgm',
#     'VERTICAL_SUM_mccnn_raw',
#     'VERTICAL_SUM_mccnn_refine',
#     'VERTICAL_SUM_sgm'
# ]
#
# #methods = [x for x in methods if 'sgm' in x]
# methods = [
#     'CB_sgm',
#     'CL_sgm',
#     'CR_sgm',
#     'CT_sgm',
#     'HORIZONTAL_FULL_sgm',
#     'VERTICAL_FULL_sgm',
#     'FULL_sgm'
# ]
#
# models = [
#     'arduino',
#     'component_0J',
#     'component_1B',
#     'component_1G',
#     'hexa_nut',
#     'hexa_screw',
#     'nodemcu',
#     'washer',
# ]
#
# # results_path = '/home/daniele/Desktop/temp/SisterResults/zzio'
# results_path = '/tmp/final_results/'
#
# target_model = 'component_1G'
# target_value = 2
# level = 2
# baseline = 2
#
# bests = []
# for method in methods:
#
#
#     results_name = target_model + "#" + method
#     filename = os.path.join(results_path, results_name + ".txt")
#
#     data = np.loadtxt(filename)
#     data = data[level * 6:level * 6 + 6, :]
#
#     rmse = np.min(data[:, target_value], axis=0)
#     i_rmse = np.argmin(data[:, target_value], axis=0)
#
#     bests.append((method, data[i_rmse], rmse))
#
# bests = sorted(bests, key=lambda x: x[1][target_value])
#
# for b in bests:
#     print("{:30}  {:.7f} {:.2f} {:000d}".format(
#         b[0],
#         b[1][target_value],
#         b[1][0],
#         int(b[1][1]),
#     ))


results_path = '/home/daniele/data/datasets/sister/v1/objects_full_scenes_results/alpha'
res = BunchOfResults(results_path)
levels = [0, 1, 3, 5]
enable_bold = False
methods =  [
'CR_sgm'
]
objects = sorted(BunchOfResults.MODELS)[:4]

level_baseline_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 5
}

models_levels_map = {
    'arduino': [0, 1],
    'nodemcu': [0, 1],
    'washer': [0, 1],
    'hexa_nut': [0, 1],
    'hexa_screw': [0, 1],
    'component_1B': [1, 2],
    'component_1G': [1, 2],
    'component_0J': [1, 2],
}

whole_data = []
for method in methods:
    local_data = []
    for model in objects:
        for level in models_levels_map[model]:
            #print(method, model, level)
            #print(res.getValue(model, method, level, -1, BunchOfResults.VALUE_RMSE)[0])
            local_data.append( res.getValue(model, method, level, 2, BunchOfResults.VALUE_RMSE)[0])
            # local_data.append((
            #     model,
            #     method,
            #     res.getValue(model, method, 1, -1, BunchOfResults.VALUE_ACC)
            # ))
    whole_data.append(local_data)


for model in objects:
    for level in models_levels_map[model]:
        print("{}_{}".format(model, level))

whole_data = np.array(whole_data)
max_map = np.argmin(whole_data,0)

table = ''
for i,method in enumerate(methods):
    table += method.replace('_','\_')
    for c in range(whole_data.shape[1]):
        if max_map[c] == i and enable_bold:
            table += "& \\textbf{"+"{:.4f}".format(whole_data[i, c])+"}"
        else:
            table += "& {:.4f}".format(whole_data[i, c])
    table += "\\\\\n"
print(table)
    # local_data = sorted(local_data, key=lambda x: x[2])
    # for d in local_data:
    #     print(d)
