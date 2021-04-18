import numpy 
import math
import random
import torch

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

  
