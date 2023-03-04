import numpy as np
from PIL import Image
import random

import torch
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F

def proc_image(image, func, **kargs):
    if isinstance(image, list) or isinstance(image, tuple):
        image = [func(img, **kargs) for img in image]
    else:
        image = func(image, **kargs)
    return image

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image)
            return image

class RandomRotate(object):
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
        print("DATA AUG: random rotate {}".format(self.angles))

    def __call__(self, image, target):
        angle = random.choice(self.angles)
        image = proc_image(image, F.rotate, angle=angle)
        target = F.rotate(target, angle=angle)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size
        print("DATA AUG: resize {}".format(self.size))

    def __call__(self, image, target):
        image = proc_image(image, F.resize, size=self.size)
        target = F.resize(target, size=self.size, interpolation=Image.NEAREST)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target

class RandomResizeCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(RandomResizeCrop, self).__init__(size, scale, ratio, interpolation)
        print("DATA AUG: RandomResizeCrop: {}".format(self))

    def __call__(self, image, target):
        i, j, h, w = self.get_params(target, self.scale, self.ratio)
        target = F.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)
        if isinstance(image, list):
            image = [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in image]
        else:
            image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
        print("DATA AUG: random flip {}".format(self.flip_prob))

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = proc_image(image, F.hflip)
            target = F.hflip(target)
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.func = T.ColorJitter(brightness, contrast, saturation, hue)
        print("DATA AUG: colorjitter")

    def __call__(self, image, target):
        image = proc_image(image, self.func)
        return image, target
        
class RandomShuffle(object):
    def __init__(self, shuffle_prob):
        self.shuffle_prob = shuffle_prob
        print("DATA AUG: random shuffle {}".format(self.shuffle_prob))

    def __call__(self, image, target):
        if random.random() < self.shuffle_prob:
            image = image[::-1]
        return image, target

class RandomCrop(object):
    def __init__(self, size, pad=False):
        self.size = size
        self.pad = pad
        print("DATA AUG: random crop {}, pad {}".format(self.size, self.pad))

    def __call__(self, image, target):
        if self.pad:
            image = pad_if_smaller(image, self.size)
            target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(target, (self.size, self.size))
        #image = F.crop(image, *crop_params)
        if isinstance(image, list):
            image = [F.crop(img, *crop_params) for img in image]
        else:
            image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = proc_image(image, F.to_tensor)
        target = (F.to_tensor(target) != 0).long()
        return image, target

class PILToTensor:
    def __init__(self, target_transform=None):
        self.target_transform = target_transform

    def __call__(self, image, target):
        #image = F.pil_to_tensor(image)
        image = proc_image(image, F.to_tensor)
        target = torch.as_tensor(np.array(target).copy(), dtype=torch.int64)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class ToPILImage(object):
    def __call__(self, image):
        image = proc_image(image, F.to_pil_image)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = proc_image(image, F.normalize, mean=self.mean, std=self.std)
        return image, target

class RevertNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def revertNormalize(self, tensor, mean, std):
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

    def __call__(self, image):
        image = proc_image(image, self.revertNormalize, mean=self.mean, std=self.std)
        return image
    

class ConcatImages(object):
    def __call__(self, image, target):
        image = torch.cat(image, dim=0)
        return image, target

class SplitImages(object):
    def __call__(self, image):
        image = torch.split(image, 3, 0)
        return image