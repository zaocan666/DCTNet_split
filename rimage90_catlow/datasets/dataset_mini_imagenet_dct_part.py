import os
import torch
import torch.utils.data as data

import os
import sys
import random
from datasets.vision import VisionDataset
from PIL import Image
import cv2
import os.path
import numpy as np
import torch
from turbojpeg import TurboJPEG
from datasets import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, \
    train_cr_mean_resized, train_cr_std_resized
from jpeg2dct.numpy import loads
from datasets.dataset_mini_imagenet_dct import pil_loader, accimage_loader, opencv_loader, default_loader, adjust_size
import pdb

class miniImageNet_part(data.Dataset):

    def __init__(self, root, split, transform=None, target_transform=None,
                 loader=default_loader, backend='opencv'):
        self.root = os.path.join(root, 'mini-imagenet', 'images')
        self.split = split
        self.loader = loader
        self.backend = backend
        assert (split == 'train' or split == 'val')
        self.csv_file = os.path.join(root, 'mini-imagenet', split + '_split.csv')
        self.images = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        with open(self.csv_file, 'r') as f:
            for i, line in enumerate(f):
                image_path = os.path.join(self.root, line.split(',')[0])
                label = line.split(',')[1][:-1]
                self.images.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = self.loader(image_path, backend='opencv', colorSpace='BGR')
        target = int(self.labels[index])

        # image = self.transform(image)

        if self.transform is not None:
            dct_y_low, dct_y_high = self.transform(image)
            return dct_y_low, dct_y_high, target

        return image, target

    def __len__(self):
        return len(self.labels)

