import os
import glob
import numpy as np
import pprint
import cv2


class CircularDataset(object):
    BASELINE_ZFILL = 2
    GLOBAL_ZFILL = 5
    DEFAULT_EXTENSION = 'png'
    NAMES = ['center', 'right', 'top', 'left', 'bottom']
    IMAGE_PREFIX = "frame_"
    POSE_PREFIX = "pose_"

    def __init__(self, path, side=5, repetitions=1, extension='png', camera=None):
        self.path = path
        if isinstance(path, str):
            self.images = sorted(glob.glob(os.path.join(self.path, "*."+extension)))
        elif isinstance(path, list):
            self.images = path
        self.images_pointer = 0
        self.camera = camera

        self.poses = self.findPosesFiles(self.images, extension)

        self.expected_images = 1 + 1 + 1 + side * 4

        self.image_map = {}
        self.pose_map = {}
        self.image_map['center'], self.pose_map['center'] = self.popImage()

        for i in range(side):
            counter = side - 1 - i
            counter_str = "bottom_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str], self.pose_map[counter_str] = self.popImage()

        self.popImage()

        for i in range(side):
            counter = i
            counter_str = "top_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str], self.pose_map[counter_str] = self.popImage()

        for i in range(side):
            counter = side - 1 - i
            counter_str = "left_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str], self.pose_map[counter_str] = self.popImage()

        self.popImage()

        for i in range(side):
            counter = i
            counter_str = "right_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str], self.pose_map[counter_str] = self.popImage()

    def findPosesFiles(self,image_list, image_extension, pose_extension='txt'):
        pose_list = image_list.copy()
        pose_list= [x.replace("."+image_extension, "."+pose_extension).replace(CircularDataset.IMAGE_PREFIX, CircularDataset.POSE_PREFIX) for x in pose_list]

        poses = []
        for p in pose_list:
            try:
                poses.append(np.loadtxt(p))
            except:
                poses.append(np.eye(4))
        return poses

    def popImage(self):
        img = self.images[self.images_pointer]
        pose = self.poses[self.images_pointer]

        print("POP", self.images_pointer, self.images[self.images_pointer],pose)
        self.images_pointer += 1
        return img, pose

    def getImage(self, name):
        img = cv2.imread(self.image_map[name])
        if self.camera is not None:
            img2 = cv2.undistort(img, self.camera.camera_matrix, self.camera.distortions)
            return img2
        return img

    def getPose(self, name):
        return self.pose_map[name]

    def getImageByIndices(self, name, baseline_index):
        if name == 'center':
            fullname = name
        else:
            fullname = name+"_"+str(baseline_index).zfill(CircularDataset.BASELINE_ZFILL)
        return self.getImage(fullname)

    def getPoseByIndices(self, name, baseline_index):
        if name == 'center':
            fullname = name
        else:
            fullname = name+"_"+str(baseline_index).zfill(CircularDataset.BASELINE_ZFILL)
        return self.getPose(fullname)

    def export(self, output_path, index=0, baseline_index=0):
        base_name = str(index).zfill(CircularDataset.GLOBAL_ZFILL)+"_"

        files_map = {}
        pose_files_map = {}
        for n in CircularDataset.NAMES:
            files_map[n] = base_name+"{}.{}".format(n, CircularDataset.DEFAULT_EXTENSION)
            pose_files_map[n] = base_name + "{}.{}".format(n, 'txt')

        for k, v in files_map.items():
            cv2.imwrite(os.path.join(output_path, v), self.getImageByIndices(k, baseline_index))
            np.savetxt(os.path.join(output_path, pose_files_map[k]), self.getPoseByIndices(k, baseline_index))
