import cv2
import numpy as np
import xmltodict
from open3d import *


class Utilities(object):

    def __init__(self):
        pass

    @staticmethod
    def loadXMLFile(filename):
        with open(filename) as fd:
            doc = xmltodict.parse(fd.read())
        return doc

    @staticmethod
    def loadRangeImage(filename, scaling_factor=1./256.):
        return cv2.imread(filename, cv2.IMREAD_ANYDEPTH) * scaling_factor

    @staticmethod
    def loadRGBImage(filename, color_code='RGB'):
        if color_code == 'BGR':
            return cv2.imread(filename)
        elif color_code == 'RGB':
            img = cv2.imread(filename)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def depthMapToPointCloud(depth, camera_matrix):

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w).astype(float), np.arange(h).astype(float))

        z = depth
        px = x - cx
        py = y - cy
        dx = z / fx
        dy = z / fy

        px = np.multiply(px, dx)
        py = np.multiply(py, dy)

        cloud = np.stack((px, py, z))
        cloud = np.swapaxes(cloud, 0, 1)
        cloud = np.swapaxes(cloud, 1, 2)
        return cloud

    @staticmethod
    def createPcd(cloud, color_image=None):
        pcd = PointCloud()
        pcd.points = Vector3dVector(cloud.reshape((-1, 3)))
        if color_image is not None:
            colors = color_image.astype(float).reshape((-1, 3)) / 255.
            pcd.colors = Vector3dVector(colors)
        return pcd


class Camera(object):

    def __init__(self, filename=None):
        if filename is not None:
            if '.txt' in filename:
                self.camera_matrix = np.loadtxt(filename)
            elif '.xml' in filename:
                doc = Utilities.loadXMLFile(filename)
                print("D"*20, doc['sister_camera']['camera']['camera_matrix'])
                self.camera_matrix = np.fromstring(doc['sister_camera']['camera']['camera_matrix'], sep=' ').reshape((3, 3))

    def getCameraMatrix(self):
        return self.camera_matrix

    def getFx(self):
        return self.camera_matrix[0, 0]

    def getCx(self):
        return self.camera_matrix[0, 2]

    def getFy(self):
        return self.camera_matrix[1, 1]

    def getCy(self):
        return self.camera_matrix[1, 2]

    def depthMapToPointCloud(self, depth):
        return Utilities.depthMapToPointCloud(depth, self.camera_matrix)


class SisterCamera(Camera):

    def __init__(self, filename=None):
        super(SisterCamera, self).__init__(filename)
        doc = Utilities.loadXMLFile(filename)
        self.baseline = float(doc['sister_camera']['baseline'])
        self.min_distance = float(doc['sister_camera']['min_distance'])
        self.max_distance = float(doc['sister_camera']['max_distance'])

    def getBaseline(self):
        return self.baseline

    def getMinDistance(self):
        return self.min_distance

    def getMaxDistance(self):
        return self.max_distance


class Reconstruction(object):

    def __init__(self, depth, rgb, camera: SisterCamera, is_disparity=True):
        self.camera = camera
        if type(depth) == str:
            self.depth = Utilities.loadRangeImage(depth)
        else:
            self.depth = depth

        if type(rgb) == str:
            if len(rgb) > 0:
                self.rgb = Utilities.loadRGBImage(rgb)
            else:
                self.rgb = None
        else:
            self.rgb = rgb

        if is_disparity:
            self.depth = self.camera.getFx() * self.camera.getBaseline() / (self.depth)
            self.depth = np.clip(self.depth, self.camera.getMinDistance(), self.camera.getMaxDistance())

        self.cloud = self.camera.depthMapToPointCloud(self.depth)

    def generatePCD(self):
        pcd = Utilities.createPcd(self.cloud, color_image=self.rgb)
        return pcd
