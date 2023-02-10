import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.jit.annotations import Dict

from model.Backbone import backbone_mtf_msf, unibackbone_fpn, mtf_backbone_msf, backbone_msf_mtf, backbone_msf
from model.Backbone import MTF


class DeepLabV3_MTF(torchvision.models.segmentation.deeplabv3.DeepLabV3):
    def __init__(self, backbone, classifier, aux_classifier, mtf):
        super(DeepLabV3_MTF, self).__init__(backbone, classifier, aux_classifier)
        self.MTF = mtf

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["t0_out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        t0 = x

        x = features["t1_out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        t1 = x

        x = self.MTF(t0, t1)
        result["out"] = x
        return result


# ResNet-18


def resnet18_mtf_msf_deeplabv3(args):
    backbone = backbone_mtf_msf('resnet18', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


def mtf_resnet18_msf_deeplabv3(args):
    backbone = mtf_backbone_msf('resnet18', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model

def resnet18_msf_mtf_deeplabv3(args):
    backbone = backbone_msf_mtf('resnet18', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


def resnet18_msf_deeplabv3_mtf(args):
    backbone = backbone_msf('resnet18', fpn_num=args.msf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    mtf = MTF(2, mode=args.mtf, kernel_size=3)
    model = DeepLabV3_MTF(backbone, classifier, aux_classifier, mtf)
    return model

# MobileNetV2

def mobilenetv2_mtf_msf_deeplabv3(args):
    backbone = backbone_mtf_msf('mobilenetv2', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


# VGG16

def vgg16bn_mtf_msf_deeplabv3(args):
    backbone = backbone_mtf_msf('vgg16_bn', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model

# ResNet-50

def resnet50_mtf_msf_deeplabv3(args):
    backbone = backbone_mtf_msf('resnet50', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model

# Swin-Transformer

def swinT_mtf_msf_deeplabv3(args):
    backbone = backbone_mtf_msf('swin_T', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model

# transfer

def resnet18_msf_deeplabv3(args):
    backbone = unibackbone_fpn('resnet18', fpn_num=args.msf)
    aux_classifier = None
    classifier = DeepLabHead(512, args.num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model