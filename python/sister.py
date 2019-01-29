import cv2
import numpy as np


class Utilities(object):

    def __init__(self):
        pass

    @staticmethod
    def loadRangeImage(filename, scaling_factor=1./256.):
        return cv2.imread(filename, cv2.IMREAD_ANYDEPTH) * scaling_factor

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


class Camera(object):

    def __init__(self, filename=None):
        if filename is not None:
            self.camera_matrix = np.loadtxt(filename)

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
