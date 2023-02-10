import torch
import torchvision
import torch.nn as nn

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.layers = nn.ModuleList(get_layers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.layers[0](x)
        f1 = self.layers[1](f0)
        f2 = self.layers[2](f1)
        f3 = self.layers[3](f2)
        f4 = self.layers[4](f3)
        return (f0, f1, f2, f3, f4)


def get_layers():
    model = torchvision.models.mobilenet_v2(pretrained=True)
    layer0 = model.features[:2]
    layer1 = model.features[2:4]
    layer2 = model.features[4:7]
    layer3 = model.features[7:14]
    layer4 = model.features[14:]
    return [layer0, layer1, layer2, layer3, layer4]
