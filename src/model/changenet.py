import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import models
from torch.hub import load_state_dict_from_url

from torchvision.models.resnet import BasicBlock, Bottleneck


class ResnetFeatures(models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResnetFeatures, self).__init__(block, layers, num_classes) 
        
    
    def set_parameter_requires_grad(self, feature_extracting=True):
        if feature_extracting:
            # Mark parameters to be freezed
            for param in self.parameters():
                param.requires_grad = not feature_extracting
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        #x = self.avgpool(x)
        #x = x.reshape(x.size(0), -1)
        #x = self.fc(x)
        return layer1, layer2, layer3, layer4

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def _resnet(base_class, arch, block, layers, pretrained, progress, **kwargs):
    model = base_class(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(base_class, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(base_class, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50(base_class, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(base_class, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
    
# Generate image from ResNet features
class DeconvNetwork(nn.Module):
    def __init__(self, num_channels_input, num_classes=11):
        super(DeconvNetwork, self).__init__()
        self.num_channels_input = num_channels_input        
        self.num_classes = num_classes
        self.gen_img = nn.Sequential(
            nn.BatchNorm2d(self.num_channels_input),
            nn.Conv2d(self.num_channels_input, self.num_classes,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.num_classes),
            # in_channels, out_channels, kernel_size, stride=1, padding=0,
            nn.ConvTranspose2d(self.num_classes, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),            
            nn.ConvTranspose2d(64, self.num_classes, 3, stride=1)
        )
    
    def forward(self, features):        
        # Reshape product for deconvolution blocks
        # After View product.shape: torch.Size([70, 3, 16, 16])
        #feat_img_gen = feat_img_gen.view(-1,self.num_channels_input,self.hidden_sqrt,self.hidden_sqrt)        
        output = self.gen_img(features)
        return output

    
class ChangeNetBranch(nn.Module):
    def __init__(self, num_classes=11):
        super(ChangeNetBranch, self).__init__()
        self.num_classes = num_classes        
        # Instantiate Resnet
        self.ResnetFeatures = resnet50(ResnetFeatures, pretrained=True)
        # Freeze Layers
        self.ResnetFeatures.set_parameter_requires_grad(feature_extracting=False)
        self.ResnetFeatures.eval()
        
        # Instantiate deconvolution blocks
        self.deconv_network_cp3 = DeconvNetwork(512, num_classes=num_classes)
        self.deconv_network_cp4 = DeconvNetwork(1024, num_classes=num_classes)
        self.deconv_network_cp5 = DeconvNetwork(2048, num_classes=num_classes)
    
    def forward(self, x):
        # Mark Resnet to be evaluation mode 
        #self.ResnetFeatures.eval()
        features_tupple = self.ResnetFeatures(x)
        _, cp3,cp4,cp5 = features_tupple
        
        # Run Deconvolution Network
        feat_cp3 = self.deconv_network_cp3(cp3)
        feat_cp4 = self.deconv_network_cp4(cp4)
        feat_cp5 = self.deconv_network_cp5(cp5)
        multi_layer_feature_map = feat_cp3, feat_cp4, feat_cp5
        return multi_layer_feature_map

    
class ChangeNet(nn.Module):
    def __init__(self, num_classes=11):
        super(ChangeNet, self).__init__() 
        self.num_classes = num_classes
        
        # Siamese Network
        self.branch_reference = ChangeNetBranch(num_classes=num_classes) 
        self.branch_test = ChangeNetBranch(num_classes=num_classes)
        
        # 1x1 Convolutions used to merge the reference/test branches
        self.FC_1_cp3 = nn.Conv2d(num_classes*2, num_classes, kernel_size=1)
        self.FC_1_cp4 = nn.Conv2d(num_classes*2, num_classes, kernel_size=1)
        self.FC_1_cp5 = nn.Conv2d(num_classes*2, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Select reference/test inputs
        input_shape = x.shape[-2:]
        reference_img,test_img = torch.split(x,3,1)
        #reference_img = x[0]
        #test_img = x[1]
        
        # Execute Branch Networks (ResNets + Deconvolutional Networks)
        feature_map_ref = self.branch_reference(reference_img)
        feature_map_test = self.branch_test(test_img)
        
        # Concatenate on the channel dimension (batch, channel, height, width)
        cp3 = torch.cat((feature_map_ref[0], feature_map_test[0]), dim=1)
        cp3 = F.interpolate(cp3, size=input_shape, mode='bilinear', align_corners=False)

        cp4 = torch.cat((feature_map_ref[1], feature_map_test[1]), dim=1)
        cp4 = F.interpolate(cp4, size=input_shape, mode='bilinear', align_corners=False)

        cp5 = torch.cat((feature_map_ref[2], feature_map_test[2]), dim=1)
        cp5 = F.interpolate(cp5, size=input_shape, mode='bilinear', align_corners=False)
        
        # Merge features from Test/Reference branches
        cp3 = self.FC_1_cp3(cp3)
        cp4 = self.FC_1_cp4(cp4)
        cp5 = self.FC_1_cp5(cp5)
        
        # Summing Branch
        sum_features = cp3 + cp4 + cp5
        
        # Use Softmax activation on the summed result
        #out = F.softmax(sum_features, dim=1)
        # We don't need softmax if we will use cross-entropy loss
        out = sum_features
        return out

def changenet(args):
    return ChangeNet(num_classes=2)
