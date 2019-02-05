import cv2
import numpy as np
import xmltodict
from open3d import *
import open3d


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
    def meshFromPointCloud(cloud, color_image=None):
        pcd = Utilities.createPcd(cloud)
        mesh = TriangleMesh()

        mesh.vertices = pcd.points
        print(mesh.vertices)
        print(mesh.triangles)

        w = cloud.shape[1]
        h = cloud.shape[0]
        size = w*h
        size_faces = (w-1)*(h-1)*2
        points = np.zeros((size, 3), float)
        colors = np.zeros((size, 3), float)
        triangles = np.zeros((size_faces, 3), np.int32)
        color_map = np.ones((h, w, 3))*100 if color_image is None else color_image

        color_map = color_map / 255.

        points_index = 0
        triangle_index = 0
        for r in range(0, cloud.shape[0], 1):
            for c in range(0, cloud.shape[1], 1):
                p0 = cloud[r, c, :]

                if r < cloud.shape[0]-1 and c < cloud.shape[1]-1:
                    p1 = cloud[r, c+1, :]
                    p2 = cloud[r+1, c, :]
                    p3 = cloud[r+1, c+1, :]

                    # n1 = np.cross(p2-p0, p1-p0)
                    # n1 = n1 / np.linalg.norm(n1)
                    # normals[triangle_index, :] = n1

                    # n2 = np.cross(p2-p1, p3-p1)
                    # n2 = n2 / np.linalg.norm(n2)
                    # normals[triangle_index+1, :] = n2

                    i0 = r * w + c
                    i1 = r * w + c + 1
                    i2 = (r+1)*w + c
                    i3 = (r+1)*w + c + 1

                    triangles[triangle_index, :] = np.array([i0, i2, i1])
                    triangles[triangle_index+1, :] = np.array([i1, i2, i3])
                    # triangles.append([i0, i1, i2])
                    triangle_index += 2

                    # triangles.append([i1, i3, i2])
                points[points_index, :] = p0
                colors[points_index, :] = color_map[r, c, :]
                points_index += 1

        mesh.triangles = open3d.Vector3iVector(np.array(triangles))
        mesh.vertices = open3d.Vector3dVector(np.array(points))
        mesh.vertex_colors = open3d.Vector3dVector(np.array(colors))
        # mesh.triangle_normals = open3d.Vector3dVector(np.array(normals))
        mesh.compute_vertex_normals()

        return mesh

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
