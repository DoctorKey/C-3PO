import os
import torch
import numpy as np
import PIL
from PIL import Image

from os.path import join as pjoin, splitext as spt

from dataset.dataset import CDDataset, get_transforms
import dataset.transforms as T

import dataset.path_config as Data_path

class PCD_CV(CDDataset):
    # all images are 256x256
    # object: white(255)  ->  True
    #                  toTensor 
    def __init__(self, root, rotation=True, transforms=None):
        super(PCD_CV, self).__init__(root, transforms)
        self.root = root
        self.rotation = rotation
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0])
                if self.rotation or idx % 4 == 0:
                    gt.append(pjoin(self.root, 'mask', file))
                    t0.append(pjoin(self.root, 't0', file.replace('png', 'jpg')))
                    t1.append(pjoin(self.root, 't1', file.replace('png', 'jpg')))
        return gt, t0, t1




class PCD_Raw(CDDataset):
    # all images are 224x1024
    # object: black(0)  ->   white(255)  ->  True
    #                 invert           toTensor  
    def __init__(self, root, num=0, train=True, transforms=None, revert_transforms=None):
        super(PCD_Raw, self).__init__(root, transforms)
        assert num in [0, 1, 2, 3, 4]
        self.root = root
        self.num = num
        self.istrain = train
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0])
                img_is_test = self.num * 2 <= (idx % 10) < (self.num + 1) * 2
                if (self.istrain and not img_is_test) or (not self.istrain and img_is_test):
                    gt.append(pjoin(self.root, 'mask', file))
                    t0.append(pjoin(self.root, 't0', file.replace('png', 'jpg')))
                    t1.append(pjoin(self.root, 't1', file.replace('png', 'jpg')))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(PCD_Raw, self).get_raw(index)
        mask = PIL.ImageOps.invert(mask)
        return imgs, mask



def get_pcd_raw(args, sub, num=0, train=True):
    assert sub in ['GSV', 'TSUNAMI']
    assert num in [0, 1, 2, 3, 4]
    root = os.path.join(Data_path.get_dataset_path('PCD_raw'), sub)
    input_size = args.input_size
    size_dict = {
        224: (224, 1024),
        256: (256, 1024),
        448: (448, 2048)
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = PCD_Raw(root, num, train, transforms=transforms, revert_transforms=revert_transforms)
    dataset.name = sub
    mode = "Train" if train else "Test"
    print("PCD_Raw_{}_{} {}: {}".format(sub, num, mode, len(dataset)))
    return dataset

def get_GSV(args, train=True):
    return get_pcd_raw(args, 'GSV', args.data_cv, train=train)

def get_TSUNAMI(args, train=True):
    return get_pcd_raw(args, 'TSUNAMI', args.data_cv, train=train)

def get_pcd_cv(args, train=True):
    num = args.data_cv
    input_size = args.input_size
    assert num in [0, 1, 2, 3, 4]
    root = os.path.join(Data_path.get_dataset_path('PCD_CV'), 'set{}'.format(num))
    assert input_size in [224, 256, 448], "input_size: 224, 256, 448"
    mode = 'train' if train else 'test'
    transforms, _ = get_transforms(args, train)
    dataset = PCD_CV(os.path.join(root, mode), transforms=transforms)
    dataset.name = 'PCD_CV_{}'.format(num)
    print("PCD_CV_{} {}: {}".format(num, mode, len(dataset)))
    return dataset


def get_pcd_cv_wo_rot(args, train=True):
    num = args.data_cv
    input_size = args.input_size
    assert num in [0, 1, 2, 3, 4]
    root = os.path.join(Data_path.get_dataset_path('PCD_CV'), 'set{}'.format(num))
    assert input_size in [224, 256, 448], "input_size: 224, 256, 448"
    mode = 'train' if train else 'test'
    transforms, _ = get_transforms(args, train)
    dataset = PCD_CV(os.path.join(root, mode), rotation=False, transforms=transforms)
    print("PCD_CV_woRot_{} {}: {}".format(num, mode, len(dataset)))
    return dataset

