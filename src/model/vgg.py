import torch
import torchvision
import torch.nn as nn

from model.backbone_base import Backbone


class VGG(Backbone):
    def __init__(self, name):
        assert name in ['vgg16', 'vgg16_bn']
        self.name = name
        super(VGG, self).__init__(get_layers(name))


def get_layers(name):
    if name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        layer0 = model.features[:5]
        layer1 = model.features[5:10]
        layer2 = model.features[10:17]
        layer3 = model.features[17:24]
        layer4 = model.features[24:]
    elif name == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=True)
        layer0 = model.features[:7]
        layer1 = model.features[7:14]
        layer2 = model.features[14:24]
        layer3 = model.features[24:34]
        layer4 = model.features[34:]
    else:
        raise ValueError(name)
    return [layer0, layer1, layer2, layer3, layer4]
