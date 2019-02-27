import cv2
import numpy as np


path = '/home/daniele/data/datasets/sister/v1/objects_full_scenes_gt/arduino_1.exr'
#path = '/home/daniele/Downloads/object_temp/snapshots/DepthBig15660.tiff'

def mouseCallback(evt,x,y,d,a):
    if 'exr' in path:
        print(img[y,x])
    else:
        disparity = img[y,x] / 256.
        depth = 500 * 0.01 / disparity
        print(depth)


cv2.namedWindow("test",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("test",mouseCallback)
#img = cv2.imread('/home/daniele/data/datasets/sister/v1/objects_full_scenes/plane/level_3_010/output/00000_classical_multiview.png', cv2.IMREAD_ANYDEPTH)
img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(float)
img = np.clip(img,0,0.1)
print(np.min(img),np.max(img))
img = (img -np.min(img))/(np.max(img) - np.min(img))
print(np.min(img),np.max(img))
cv2.imshow("img",img)
cv2.waitKey(0)

# print(np.min(img),np.max(img))
# if 'tiff' in path or 'exr' in path:
#     depth = np.clip(img,0,0.9)
# else:
#     depth = 1. /(500 * 0.01 / img)
#     print(depth)
# print(np.min(depth),np.max(depth))
# normalized_depth = (depth - np.min(depth))/(np.max(depth)-np.min(depth))
# cv2.imshow("test", depth)
# cv2.waitKey(0)
# cv2.imwrite("/tmp/gippo.jpg",(normalized_depth*255).astype(np.uint8))