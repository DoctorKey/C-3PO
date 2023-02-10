import os
import torch
import numpy as np
from torch.utils.data import Dataset
import PIL
from PIL import Image

from os.path import join as pjoin, splitext as spt

import dataset.transforms as T 
from torchvision.transforms import functional as F

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



class CDDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(CDDataset, self).__init__()
        self.root = root
        self.gt, self.t0, self.t1 = [], [], []
        self._transforms = transforms
        self._revert_transforms = None
        self.name = ''
        self.num_classes = 2

    def _check_validness(self, f):
        return any([i in spt(f)[1] for i in ['jpg','png']])

    def _pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _init_data_list(self):
        pass

    def get_raw(self, index):
        fn_t0 = self.t0[index]
        fn_t1 = self.t1[index]
        fn_mask = self.gt[index]

        img_t0 = self._pil_loader(fn_t0)
        img_t1 = self._pil_loader(fn_t1)
        imgs = [img_t0, img_t1]

        mask = self._pil_loader(fn_mask).convert("L")
        return imgs, mask

    def __getitem__(self, index):
        imgs, mask = self.get_raw(index)
        if self._transforms is not None:
            imgs, mask = self._transforms(imgs, mask)
        return imgs, mask

    def __len__(self):
        return len(self.gt)

    def get_mask_ratio(self):
        all_count = 0
        mask_count = 0
        for i in range(len(self.gt)):
            _, mask = self.get_raw(i)
            target = (F.to_tensor(mask) != 0).long()
            mask_count += target.sum()
            all_count += target.numel()
        mask_ratio = mask_count / float(all_count)
        background_ratio = (all_count - mask_count) / float(all_count)
        return [mask_ratio, background_ratio]

    def get_pil(self, imgs, mask, pred=None):
        assert self._revert_transforms is not None
        t0, t1 = self._revert_transforms(imgs.cpu())
        w, h = t0.size
        output = Image.new('RGB', (w * 2, h * 2))
        output.paste(t0)
        output.paste(t1, (w, 0))
        mask = F.to_pil_image(mask.cpu().float())
        output.paste(mask, (0, h))
        pred = F.to_pil_image(pred.cpu().float())
        output.paste(pred, (w, h))
        return output

def get_transforms(args, train, size_dict=None):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    if size_dict is not None:
        assert args.input_size in size_dict, "input_size: {}".format(size_dict.keys())
        input_size = size_dict[args.input_size]
    else:
        input_size = args.input_size

    mode = "Train" if train else "Test"
    print("{} Aug:".format(mode))
    augs = []
    if train:
        if args.randomcrop:
            if args.input_size == 256:
                augs.append(T.Resize(286))
                augs.append(T.RandomCrop(input_size))
            elif args.input_size == 512:
                augs.append(T.Resize((572, 572)))
                augs.append(T.RandomCrop(512))
            else:
                raise ValueError(args.input_size)
        else:
            augs.append(T.Resize(input_size))
        augs.append(T.RandomHorizontalFlip(args.randomflip))
    else:
        augs.append(T.Resize(input_size))
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    augs.append(T.ConcatImages())
    transforms = T.Compose(augs)
    revert_transforms = T.Compose([
        T.SplitImages(),
        T.RevertNormalize(mean=mean, std=std),
        T.ToPILImage()
    ])
    return transforms, revert_transforms