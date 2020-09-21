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
import pdb

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def opencv_loader(path, colorSpace='YCrCb'):
    image = cv2.imread(str(path))
    # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets/cvtransforms/test/raw.jpg', image)
    if colorSpace == "YCrCb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets/cvtransforms/test/ycbcr.jpg', image)
    elif colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def default_loader(path, backend='opencv', colorSpace='YCrCb'):
    from torchvision import get_image_backend
    if backend == 'opencv':
        return opencv_loader(path, colorSpace=colorSpace)
    elif get_image_backend() == 'accimage' and backend == 'acc':
        return accimage_loader(path)
    elif backend == 'pil':
        return pil_loader(path)
    else:
        raise NotImplementedError

def adjust_size(y_size, cbcr_size):
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size

class miniImageNet(data.Dataset):

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
            dct_y, dct_cb, dct_cr = self.transform(image)

        if self.backend == 'dct':
            if dct_cb is not None:
                image = torch.cat((dct_y, dct_cb, dct_cr), dim=1)
                return image, target
            else:
                return dct_y, target
        else:
            if dct_cb is not None:
                return dct_y, dct_cb, dct_cr, target
            else:
                return dct_y, target

        return image, target

    def __len__(self):
        return len(self.labels)

