import os
import csv
import zipfile
import torch
from munch import Munch
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from tqdm import tqdm
import logging

import pytorch_lightning as pl

from pl_bolts.models.self_supervised import SimSiam, SwAV, SimCLR, BYOL
from pl_bolts.datamodules import ImagenetDataModule, STL10DataModule, CIFAR10DataModule
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform, SimCLRFinetuneTransform)
from pl_bolts.models.self_supervised.swav.transforms import (SwAVTrainDataTransform, SwAVEvalDataTransform, SwAVFinetuneTransform)
from pl_bolts.models.self_supervised import SSLFineTuner
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization, stl10_normalization
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

from pytorch_lightning.metrics.functional import accuracy

import torch.nn.functional as F
from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from exdark_datamodule import SSLExDarkDataModule


args = {
        'dataset': 'exdark',
        'ckpt_path': '/home/data/ckpt',
        'data_dir': '/home/data/ExDark',
        'mask_dir': '/home/data/ExDark_Annno', # I mistakenly named it Annno
        'batch_size': 256,
        'num_workers': 0,
        'gpus': 8,
        'in_features': 2048,
        'dropout': 0.,
        'learning_rate': 0.1,
        'weight_decay': 1e-4,
        'nesterov': False,
        'scheduler_type': 'cosine',
        'gamma': 0.1,
        'final_lr': 0.,
        'epochs': 10,
}

args = Munch(args)

pl.seed_everything(224)

def main(config):
    # dataset 
    dm = SSLExDarkDataModule()
    
    # 1) load from pretrained weight
    simclr_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    swav_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    simclr = SimCLR.load_from_checkpoint(simclr_weight_path, strict=False)
    swav = SwAV.load_from_checkpoint(swav_weight_path, strict=True)

    simclr.freeze()
    swav.freeze()

    # 2) SSL Finetuner
    simclr_tuner = SSLFineTuner(
            simclr, 
            in_features=simclr.feat_dim,
            num_classes=dm.num_classes,
            epochs=args.epochs,
            hidden_dim=None,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            scheduler_type=args.scheduler_type,
            gamma=args.gamma,
            final_lr=args.final_lr
    )

    swav_tuner = SSLFineTuner(
             swav, 
             in_features=swav.feat_dim, 
             num_classes=dm.num_classes,
             epochs = args.epochs, 
             hidden_dim=None, 
             dropout=args.dropout,
             learning_rate = args.learning_rate,
             weight_decay=args.weight_decay,
             nesterov=args.nesterov,
             scheduler_type=args.scheduler_type,
             gamma=args.gamma,
             final_lr=args.final_lr
    )

    # 3) train the fine tuner than test 
    simclr_trainer = pl.Trainer(
            gpus=args.gpus,
            num_nodes=1,
            max_epochs=args.epochs,
            distributed_backend='ddp',
            sync_batchnorm=True if args.gpus > 1 else False,
    )
    swav_trainer = pl.Trainer(
            gpus=args.gpus, 
            num_nodes=1,
            max_epochs=args.epochs,
            distributed_backend='ddp',
            sync_batchnorm=True if args.gpus > 1 else False,
    )

    simclr_trainer.fit(simclr_tuner, dm)
    simclr_trainer.test(datamodule=dm)

    swav_trainer.fit(swav_trainer, dm)
    swav_trainer.test(datamodule=dm)


if __name__=="__main__":
    main(args)
