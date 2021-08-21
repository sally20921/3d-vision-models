import os
import zipfile
import torch
from munch import Munch

import pytorch_lightning as pl

from pl_bolts.models.self_supervised import SimSiam, SwAV
from pl_bolts.datamodules import ImagenetDataModule, STL10DataModule
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform)
from pl_bolts.models.self_supervised.swav.transforms import (SwAVTrainDataTransform, SwAVEvalDataTransform, SwAVFinetuneTransform)
from pl_bolts.models.self_supervised import SSLFineTuner
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization, stl10_normalization

from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datasets import UnlabeledImagenet

# http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_devkit_t12.tar.gz
# http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar
# wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar
args = {
        'dataset': 'imagenet',
        'meta_path': '/home/data',
        'ckpt_path': '/home/data/ckpt',
        'data_dir': '/home/data',
        'batch_size': 512,
        'num_workers': 8,
        'gpus': 8,
        'in_features': 2048,
        'dropout': 0.,
        'learning_rate': 0.3,
        'weight_decay': 1e-6,
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
    UnlabeledImagenet.generate_meta_bins(config.meta_path)

if __name__=="__main__":
    main(args)
