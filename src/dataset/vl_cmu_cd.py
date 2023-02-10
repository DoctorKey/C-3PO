import os

from os.path import join as pjoin, splitext as spt

import dataset.transforms as T 
from dataset.dataset import CDDataset, get_transforms

import dataset.path_config as Data_path

class VL_CMU_CD(CDDataset):
    # all images are 512x512
    def __init__(self, root, rotation=True, transforms=None, revert_transforms=None):
        super(VL_CMU_CD, self).__init__(root, transforms)
        self.root = root
        self.rotation = rotation
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0].split('_')[-1])
                if self.rotation or idx == 0:
                    gt.append(pjoin(self.root, 'mask', file))
                    t0.append(pjoin(self.root, 't0', file))
                    t1.append(pjoin(self.root, 't1', file))
        return gt, t0, t1



def get_VL_CMU_CD(args, train=True):
    mode = 'train' if train else 'test'
    raw_root = Data_path.get_dataset_path('CMU_binary')
    size_dict = {
        512: (512, 512),
        768: (768, 1024)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = VL_CMU_CD(os.path.join(raw_root, mode), 
        transforms=transforms, revert_transforms=revert_transforms)
    print("VL_CMU_CD {}: {}".format(mode, len(dataset)))
    return dataset
        
class VL_CMU_CD_Raw(CDDataset):
    # all images are 1024x768
    def __init__(self, root, transforms=None, revert_transforms=None):
        super(VL_CMU_CD_Raw, self).__init__(root, transforms)
        self.root = root
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        sub_class = list(f for f in os.listdir(self.root) if os.path.isdir(pjoin(self.root, f)))
        for c in sub_class:
            img_root = pjoin(self.root, c, 'RGB')
            mask_root = pjoin(self.root, c, 'GT')
            for f in os.listdir(mask_root):
                if self._check_validness(f):
                    gt.append(pjoin(mask_root, f))
                    t0.append(pjoin(img_root, f.replace("gt", "1_")))
                    t1.append(pjoin(img_root, f.replace("gt", "2_")))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(VL_CMU_CD_Raw, self).get_raw(index)
        # x == 255 is sky
        mask = mask.point(lambda x: int(0 < x < 255) * 255)
        return imgs, mask


def get_VL_CMU_CD_Raw(args, train=True):
    mode = 'train' if train else 'test'
    raw_root = Data_path.get_dataset_path('CMU_raw')
    size_dict = {
        512: (512, 512),
        768: (768, 1024)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = VL_CMU_CD_Raw(os.path.join(raw_root, mode), 
        transforms=transforms, revert_transforms=revert_transforms)
    print("VL_CMU_CD_Raw {}: {}".format(mode, len(dataset)))
    return dataset
        


