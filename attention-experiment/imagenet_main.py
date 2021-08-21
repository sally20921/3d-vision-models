import os
import zipfile
import torch
from munch import Munch

import pytorch_lightning as pl

from pl_bolts.models.self_supervised import SimSiam, SwAV
from pl_bolts.datamodules import ImagenetDataModule, STL10DataModule, CIFAR10DataModule
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform)
from pl_bolts.models.self_supervised.swav.transforms import (SwAVTrainDataTransform, SwAVEvalDataTransform, SwAVFinetuneTransform)
from pl_bolts.models.self_supervised import SSLFineTuner
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization, stl10_normalization

from pytorch_lightning.metrics.functional import accuracy

args = {
        'dataset': 'imagenet',
        'ckpt_path': '/home/data/ckpt',
        'data_dir': '/home/data/imagenet',
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
    if config.dataset == 'imagenet':
        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        dm.train_transforms = SwAVFinetuneTransform(
                normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=False
        )
        dm.val_transformations = SwAVFinetuneTransform(
                normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        dm.test_transforms = SwAVFinetuneTransform(
                normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        args.num_samples = 1
        args.maxpool1 = True
        args.first_conv = True
    elif config.dataset == 'imagenet':
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
        args.num_samples = 0

        dm.train_transforms = SwAVFinetuneTransform(
            normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=False
        )
        dm.val_transforms = SwAVFinetuneTransform(
            normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        dm.test_transforms = SwAVFinetuneTransform(
            normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        args.maxpool1 = False
        args.first_conv = True
    else:
        raise NotImplementedError
        

    
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    swav = SwAV.load_from_checkpoint(weight_path, strict=True)
    tuner = SSLFineTuner(
             swav, 
             in_features=args.in_features, 
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
    trainer = pl.Trainer(
            gpus=args.gpus,
            num_nodes=1,
            precision=16,
            max_epochs=args.epochs,
            distributed_backend='ddp',
            sync_batchnorm=True if args.gpus > 1 else False,
    )

    
    
    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)


if __name__=="__main__":
    main(args)
