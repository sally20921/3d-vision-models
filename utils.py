import numpy 
import math
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pylot as plt

fx = fy = 721.5
cx = 690.5
xy = 172.8
baseline = 0.54

fx = 640
cy = 640
ppx = 320
ppy = 320
h = 640
w = 640

cam_intr = np.array([351, 351, 320, 240], dtype=np.float64)
img_size = np.array([480, 640], dtype=np.int32)

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1-pt2)
        side_b = np.linalg.norm(pt2-pt3)
        side_c = np.linalg.norm(pt3-pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s-side_a) * (s-side_b) * (s-side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))


def pyntcloud_read_off(filename):
    with open(filename) as off:
        first_line = off.readline()
        if "OFF" not in first_line:
            raise ValueError("The file does not start with the word OFF")

        color = True if "C" in first_line else False

        count = 1
        for line in off:
            count+=1 
            if line.startswith("#"):
                continue
            line = line.strip().split()
            if len(line) > 1:
                n_points = int(line[0])
                n_faces = int(line[1])
                break

        data = {}
        point_names = ["x", "y", "z"]
        if color:
            point_names.extend(["red", "green", "blue"])

        data["points"] = pd.read_csv(filename, sep=" ", header=None,
                engine="python", skiprows=count, skipfooter=n_faces,
                names=point_names, index_col=False)

        for n in ["x", "y", "z"]:
            data["points"][n] = data["points"][n].astype(np.float32)

        if color:
            for n in ["red", "green", "blue"]:
                data["points"][n] = data["points"][n].astype(np.float32)

        data["mesh"] = pd.read_csv(filename, sep=" ", header=None,
                engine="python", skirows=(count+n_points), usecols=[1,2,3],
                names=["v1", "v2", "v3"])
        return data

def convert_pointcloud_to_depth(pointcloud, camera_intrinsic):
    x_ = pointcloud["points"]["x"]
    y_ = pointcloud["points"]["y"]
    z_ = pointcloud["points"]["z"]

    m = x_[np.nonzero(z_)] / z_[np.nonzero(z_)]
    n = y_[np.nonzero(z_)] / z_[np.nonzero(z_)]

    x = m * camera_intrinsic.fx + camera_intrinsic.ppx
    y = n * camera_intrinsic.fy + camera_intrinsic.ppy

    return x, y

def project_pointcloud(pointcloud, img, fx, fy, cx ,cy):
    imgToReturn = np.zeros(img.shape, dtype=np.uint8)
    for point in pointcloud:
        X = point[0]
        Y = point[1]
        Z = point[2]
        i = np.floor((X*fx / Z) + cx)
        j = np.floor((Y*fy / Z) + cy)
        if i < imgToReturn.shape[0] and j < imgToReturn.shape[1]:
            imgToReturn[i, j] = np.array([point[5], point[4], point[3]])
    return imgToReturn
