import os,glob,cv2
import numpy as np
from sister.transformations import *
from sister.sister import  Camera

camera = Camera('/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')
folder = '/home/daniele/Desktop/temp/RobotStereoExperiments/Datasets/plate_multi_height/plate_rgb/subset_3_baseline_4'

original_image_path = os.path.join(folder, "00000_center.png")
marker_pose_path = os.path.join(folder, "00000_center.png_detection.txt")
synth_image_path =  os.path.join(folder, "synth/Image/Image.png")

original_image = cv2.imread(original_image_path)
synth_image = cv2.imread(synth_image_path)


pose_raw = np.loadtxt(marker_pose_path)[1, :]
print("POSE RAW",pose_raw)

q = pose_raw[3:]
T = quaternion_matrix(q)
T[:3,3] = pose_raw[:3].reshape((1,3))
print(T)

T_marker = T
Rvec, _= cv2.Rodrigues(np.eye(3))
Tvec = np.array([0,0,0],float)
print(Rvec,Tvec)

points = np.array([
    [0.05, 0.05, 0,1],
    [-0.05, -0.05, 0,1],
])
points = np.matmul(T_marker, points.T).T[:,:3]
print(points)
#projected, _ = cv2.projectPoints(points,Rvec,Tvec,camera.camera_matrix,camera.distortions)

projected = np.matmul(camera.camera_matrix, points.T).T
print(projected)

for p in projected:
    p = p.ravel()
    p = p / p[2]
    p = p[:2]
    print("P",p)
    print(tuple(p.astype(int)))
    cv2.circle(synth_image,tuple(p.astype(int)),5,(55,0,255),2)
    cv2.circle(original_image, tuple(p.astype(int)), 5, (55, 0, 255), 2)
    print(p)
cv2.imshow("image",original_image)
cv2.imshow("synth", synth_image)
cv2.waitKey(0)



