import os
from path import Path
import numpy as np
import math
import random
import torch
import utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def default_transforms():
    return transforms.Compose([
                        utils.ToTensor()
                    ])

def ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transforms=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform is not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    print("reading file ", file)
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = utils.read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

if __name__ == "__main__":
    root_dir = '/home/data/seri/ModelNet40'
    folders = [dir for dir in sorted(os.listdir(root_dir))]
    print(folders)
