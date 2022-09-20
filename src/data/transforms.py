import random

import numpy as np
from scipy.ndimage import gaussian_filter


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
