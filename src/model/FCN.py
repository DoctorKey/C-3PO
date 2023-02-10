from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.jit.annotations import Dict

from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCN, FCNHead

from model.Backbone import unibackbone_fpn, backbone_mtf_msf, bibackbone_mtf_msf, bibackbone_mtf_msf


def resnet18_mtf_msf_fcn(args):
    backbone = backbone_mtf_msf('resnet18', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model


# symmetry

def biresnet18_mtf_msf_fcn(args):
    backbone = bibackbone_mtf_msf('resnet18', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model

# transfer

def resnet18_msf_fcn(args):
    backbone = unibackbone_fpn('resnet18', fpn_num=args.msf)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model



