import random
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.ndimage import gaussian_filter

from .datasets import SyntheticDataset


class TAVRDataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset_name: str,
                 dataset_kwargs: dict,
                 phase: str = 'final',
                 batch_size: Union[int, str] = 8,
                 num_workers: int = 4):
        super().__init__()
        self.dataset_cls = globals()[dataset_name]
        self.dataset_kwargs = dataset_kwargs
        self.phase = phase
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = Compose([
            RandomFlip(),
            GaussianBlur(),
            StandardizeImage()
        ])
        self.test_transforms = StandardizeImage()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.phase == 'tuning':
                self.trainset = self.dataset_cls(**self.dataset_kwargs,
                                                 split='train',
                                                 transforms=self.train_transforms)
                self.valset = self.dataset_cls(**self.dataset_kwargs,
                                               split='val',
                                               transforms=self.test_transforms)
            else:
                self.trainset = self.dataset_cls(**self.dataset_kwargs,
                                                 split='trainval',
                                                 transforms=self.train_transforms)

        if stage == 'test' or stage is None:
            self.testset = self.dataset_cls(**self.dataset_kwargs,
                                            split='test',
                                            transforms=self.test_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

    def val_dataloader(self):
        if self.phase == 'tuning':
            return torch.utils.data.DataLoader(self.valset,
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                pin_memory=True)
        else:
            return None

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class StandardizeImage:
    def __call__(self, sample):
        if sample['modes'] in [3, 5, 6, 7]:  # image missing
            return sample

        image = sample['image']
        if len(image.shape) > 2:
            image = (image - image.mean()) / image.std()
            sample['image'] = image
        return sample


class GaussianBlur:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if sample['modes'] in [3, 5, 6, 7]:  # image missing
            return sample

        img = sample['image']

        if len(img.shape) > 2:
            if random.random() < self.p:
                sigma = random.uniform(0.1, 0.9)
                img = gaussian_filter(img, sigma)
            sample['image'] = img
        return sample


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if sample['modes'] in [3, 5, 6, 7]:  # image missing
            return sample

        img = sample['image']

        if len(img.shape) > 2:
            if random.random() < self.p:
                img = np.flip(img, axis=0)
            if random.random() < self.p:
                img = np.flip(img, axis=1)
            if random.random() < self.p:
                img = np.flip(img, axis=2)
            sample['image'] = img
        return sample
