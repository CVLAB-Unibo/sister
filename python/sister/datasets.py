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

    def __init__(self, path, side=5, repetitions=1, extension='png'):
        self.path = path
        self.images = sorted(glob.glob(os.path.join(self.path, "*."+extension)))
        self.images_pointer = 0

        self.expected_images = 1 + 1 + 1 + side * 4

        self.image_map = {}
        self.image_map['center'] = self.popImage()

        for i in range(side):
            counter = side - 1 - i
            counter_str = "bottom_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str] = self.popImage()

        self.popImage()

        for i in range(side):
            counter = i
            counter_str = "top_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str] = self.popImage()

        for i in range(side):
            counter = side - 1 - i
            counter_str = "left_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str] = self.popImage()

        self.popImage()

        for i in range(side):
            counter = i
            counter_str = "right_"+str(counter).zfill(CircularDataset.BASELINE_ZFILL)
            self.image_map[counter_str] = self.popImage()

    def popImage(self):
        img = self.images[self.images_pointer]
        print("POP", self.images_pointer, self.images[self.images_pointer])
        self.images_pointer += 1
        return img

    def getImage(self, name):
        return cv2.imread(self.image_map[name])

    def getImageByIndices(self, name, baseline_index):
        if name == 'center':
            fullname = name
        else:
            fullname = name+"_"+str(baseline_index).zfill(CircularDataset.BASELINE_ZFILL)
        return self.getImage(fullname)

    def export(self, output_path, index=0, baseline_index=0):
        base_name = str(index).zfill(CircularDataset.GLOBAL_ZFILL)+"_"

        files_map = {}
        for n in CircularDataset.NAMES:
            files_map[n] = base_name+"{}.{}".format(n, CircularDataset.DEFAULT_EXTENSION)

        for k, v in files_map.items():
            cv2.imwrite(os.path.join(output_path, v), self.getImageByIndices(k, baseline_index))
