import numpy as np
import cv2
import os, glob
from fpdf import FPDF
from sister.datasets import BunchOfResults

results_path = '/tmp/final_results/'
dataset_path = '/home/daniele/data/datasets/sister/v1/objects_full_scenes/'

res = BunchOfResults(results_path, dataset_path)

pdf = FPDF()

level = 0
baseline = 2
h = 480
w = 640
padding = 20
margin = 20
pdf.add_page()

OBJECT_LEVELS ={
    'arduino': 1,
    'nodemcu': 1,
    'washer': 0,
    'hexa_nut': 0,
    'hexa_screw': 0,
    'component_1B': 1,
    'component_1G': 1,
    'component_0J': 1,
}

OBJECT_BASELINES ={
    'arduino': 10,
    'nodemcu': 10,
    'washer': 2,
    'hexa_nut': 2,
    'hexa_screw': 2,
    'component_1B': 10,
    'component_1G': 10,
    'component_0J': 10,
}

OBJECT_COLORMAP ={
    'arduino': '(img * 1.5).astype(np.uint8)  + 30',
    'nodemcu': '(img * 1.5).astype(np.uint8)  + 30',
    'washer': 'img * 2 + 30',
    'hexa_nut': 'img * 2 + 30',
    'hexa_screw': '(img * 3.5).astype(np.uint8) + 30',
    'component_1B': 'img * 2 + 30',
    'component_1G': 'img * 2 + 30',
    'component_0J': 'img * 2 + 30',
}

METHODS_SUBSET = [
    'CR_sgm',
    'FULL_sgm',
    'FULL_mccnn_raw'
]

rows = len(BunchOfResults.MODELS)
cols = 1 + len(METHODS_SUBSET)
wall = np.ones((padding + rows*(h+margin), (padding + cols*(w+margin)), 3), np.uint8)*255

for i, model in enumerate(BunchOfResults.MODELS):
    local_data = []
    y = padding + i * (h + margin)
    x = padding
    level = OBJECT_LEVELS[model]
    baseline = OBJECT_BASELINES[model]
    rgb = cv2.imread(res.getRgbPath(model, level, baseline))
    wall[y:y + h, x:x + w, ::] = rgb
    for j, method in enumerate(METHODS_SUBSET):
        x = padding + (j + 1) * (w + margin)
        img = cv2.imread(res.getDepthPath(model, method, level, baseline))
        # print("MINNNY",np.min(img),np.max(img))
        # img = 255 * ( (img - np.min(img))/(np.max(img) - np.min(img)))
        # img = img.astype(np.uint8)
        # print("MANNNY",np.min(img), np.max(img))
        img = cv2.applyColorMap(eval(OBJECT_COLORMAP[model]),2)
        wall[y:y+h, x:x+w, ::] = img
        print(np.min(img),np.max(img))
        print(res.getDepthPath(model, method, 0, 2.0))

cv2.namedWindow("wall",cv2.WINDOW_NORMAL)
cv2.imshow("wall",wall)
cv2.waitKey(0)
cv2.imwrite("/tmp/matricione.png", wall)