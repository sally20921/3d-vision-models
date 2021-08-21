import os
import csv
import zipfile
import gzip
import hashlib
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from munch import Munch
import numpy as np
from torch._six import PY3

import logging 
import os
from typing import Any, Callable, Optional
import urllib.request
from abc import ABC
from typing import Sequence, Tuple, Callable, Optional
from urllib.error import HTTPError
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib


class ExDarkDataset(Dataset):
    def __init__(self, root, split: str='train'):

        
        root = self.root = os.path.expanduser(root)
        self.root = os.path.join(root, 'ExDark')
        self.meta_root = os.path.join(root, 'ExDark_Annno')

        #self.classes
        self.img_paths = []
        self.target_paths = []
        self.classes = {'Bicycle':0, 'Boat':1, 'Bottle':2, 'Bus':3, 'Car':4, 'Chair':5, 'Cup':6, 
                'Dog':7, 'Motorbike':8, 'People':9, 'Table':10}

        for _class in self.classes.keys():
            class_path = Path(os.path.join(self.root, _class))
            print("class path: ", class_path)
            meta_class_path = Path(os.path.join(self.meta_root, _class))
            sample_paths = class_path.glob('*')
            label_paths = meta_class_path.glob('*')
            for sample_path in sample_paths:
                self.img_paths.append(sample_path)
            for label_path in label_paths:
                self.target_paths.append(label_path)

        self.paths = []
        cnt = 0
        for img_path, target_path in zip(self.img_paths, self.target_paths):
            #self.paths[img_path] = target_path
            self.paths.append((img_path, target_path))
            cnt+=1

        np.random.seed(1234)
        #np.random.shuffle(self.paths)
        
        original_split = split
        if split == 'train' or split == 'val':
            split = 'train'
        if split == 'test':
            split = 'val'

        self.split = split

        # partition trainset into [train, val]
        if split == 'train':
            train, val = self.partition_train_set(self.paths, 0.8)
            if original_split == 'train':
                self.imgs = train
            if original_split == 'val':
                self.imgs = val

        self.wnids = self.classes.keys()
        self.wnids_to_idx = self.classes

        #self.samples = self.imgs
        self.targets = [s[1] for s in self.paths]
        #print(self.imgs)
        print(self.targets)

    

    def partition_train_set(self, imgs, ratio):
        size = len(imgs)
        train_size = int(size * 0.8)
        val_size = size - train_size

        val = []
        train = []

        #cts = {x: 0 for x in range(len(self.classes))}
        cnt = 0
        for idx in range(len(imgs)):
            #img_path = imgs[idx][0]
            img_path, idx_path = imgs[idx]
            if (cnt < train_size):
                train.append((img_path, idx_path))
            else:
                val.append((img_path, idx_path))
            cnt += 1

        return train, val

    def __getitem__(self, idx):
        img_path = self.paths[idx][0]
        target_path = self.paths[idx][1]
        img = Image.open(img_path).convert('RGB')
        tensor_img = torchvision.transforms.ToTensor()(img)
        labels = []
        with open(target_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                label = line.split(' ')[0]
                labels.append(self.classes(label))
        print("labels for this image is", labels)

        return {'img': img, 'labels': labels}


    def __len__(self):
        return len(self.imgs)


class SSLExDarkDataModule(LightningDataModule):
    name = 'exdark'
    def __init__(self, 
            data_dir: str = '/home/data/ExDark',
            meta_dir: Optional[str] = '/home/data/ExDark_Annno',
            num_workers: int = 4,
            batch_size: int = 256,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
        ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 10

    def train_dataloader(self, add_normalize:bool = False) -> DataLoader:
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transform

        dataset = ExDarkDataset('/home/data', split='train')
        loader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                drop_last=self.drop_last, pin_memory=self.pin_memory)

        return loader

    def val_dataloader(self, add_normalize: bool = False) -> DataLoader:
        transforms = self._default_transforms if self.val_transforms is None else self.val_transforms

        dataset = ExDarkDataset('/home/data', split='val')
        loader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                drop_last=self.drop_last, pin_memory=self.pin_memory)

    def test_dataloader(self, add_normalize: bool = False) -> DataLoader:
        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = ExDarkDataset('/home/data', split='test')
        loader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=self.drop_last,
                pin_memory=self.pin_memory)

    def _default_transforms(self) -> Callable:
        tensor_transforms = transform_lib.Compose([transform_lib.ToTensor(), ])
        return tensor_transform

def main():
    train_dataset = ExDarkDataset(root='/home/data', split='train')
    val_dataset = ExDarkDataset(root='/home/data', split='val')
    test_dataset = ExDarkDataset(root='/home/data', split='test')

    dm = SSLExDarkDataModule()
    train_dataloader = dm.train_dataloader
    val_dataloader =dm.val_dataloader
    test_dataloader = dm.test_dataloader



if (__name__ == "__main__"):
    main()
    
        




