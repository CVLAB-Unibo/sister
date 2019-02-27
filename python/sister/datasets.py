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
    NAMES_FLOW = ['center', 'bottom', 'center', 'top', 'left', 'center', 'right']
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

    @staticmethod
    def getCorrespondingPose(image_path):
        ext = os.path.splitext(image_path)[1]
        return image_path.replace(ext, ".txt").replace(CircularDataset.IMAGE_PREFIX, CircularDataset.POSE_PREFIX)


class CircularFrame(object):
    NAMES = ['center','left','right','top','bottom']

    def __init__(self, path, extension = "png"):
        self.path = path
        self.images = sorted(glob.glob(os.path.join(path, '*.' + extension)))[:5]

        self.images_map = {}
        self.poses = {}

        for image_path in self.images:
            for n in CircularFrame.NAMES:
                if n in image_path:
                    self.images_map[n] = image_path
                    self.poses[n] = np.loadtxt(image_path.replace("."+extension, ".txt"))

        if len(self.images_map) != 5:
            print("Frame error in size!", self.images_map)
            import sys
            sys.exit(0)

    def getImage(self, name):
        if name in self.images_map:
            return cv2.cvtColor(cv2.imread(self.images_map[name]), cv2.COLOR_BGR2RGB)
        return None

    def getPose(self,name):
        if name in self.poses:
            return self.poses[name]
        else:
            return np.eye(4)

    def baseline(self):
        return self.computeDistance('center', 'bottom')

    def computeDistance(self, n1,n2):
        p1 = self.poses[n1][:3,3]
        p2 = self.poses[n2][:3,3]
        return np.linalg.norm(p1- p2)




class ScaleManager(object):

    @staticmethod
    def getScaleByName(source_name):
        tags_scale_map = {
            "classical": 256.,
            "mccnn": 1.,
            "sgm": 1.
        }
        for name, s in tags_scale_map.items():
            if name in source_name:
                scale = s
        return scale



class BunchOfResults(object):
    VALUE_RMSE = 4
    VALUE_MSE = 3
    VALUE_MAE = 2
    VALUE_ACC = 6


    MODELS = [
        'component_0J',
        'component_1B',
        'component_1G',
        'arduino',
        'nodemcu',
        'hexa_nut',
        'hexa_screw',
        'washer',
    ]
    METHODS = [
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

    METHODS_SGM = [
        'CB_sgm',
        'CL_sgm',
        'CR_sgm',
        'CT_sgm',
        'FULL_sgm',
        # 'HORIZONTAL_FULL_sgm',
        # 'HORIZONTAL_SUM_sgm',
        # 'SUM_sgm',
        # 'VERTICAL_FULL_sgm',
        # 'VERTICAL_SUM_sgm'
    ]

    def __init__(self, path, dataset_path=''):
        self.path = path
        self.dataset_path = dataset_path
        self.models_map = {}
        for model in BunchOfResults.MODELS:
            if model not in self.models_map:
                self.models_map[model] = {}
            for method in BunchOfResults.METHODS:
                results_name = model + "#" + method
                filename = os.path.join(self.path, results_name + ".txt")
                try:
                    self.models_map[model][method] = np.loadtxt(filename)
                except:
                    self.models_map[model][method] = None

    def getDepthPath(self, model, method, level, baseline):
        baseline = str(int(baseline)).zfill(3)
        subfolder_name = "level_{}_{}".format(int(level), baseline)
        path = os.path.join(self.dataset_path, model, subfolder_name,'output', method+".png")
        return path

    def getRgbPath(self, model, level, baseline):
        baseline = str(int(baseline)).zfill(3)
        subfolder_name = "level_{}_{}".format(int(level), baseline)
        path = os.path.join(self.dataset_path, model, subfolder_name, '00000_center.png')
        return path

    def getValue(self, model, method, level, baseline_index, target_value):
        data = self.models_map[model][method]
        data = data[level*6:level*6 + 6]

        v = data[:,target_value]
        if baseline_index >= 0:
            v = v[baseline_index]
        else:
            v = np.min(v)
            baseline_index = np.argmin(v)
        return v, data[baseline_index, 1]


