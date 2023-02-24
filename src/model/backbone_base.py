import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, layer_list):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.layers[0](x)
        f1 = self.layers[1](f0)
        f2 = self.layers[2](f1)
        f3 = self.layers[3](f2)
        f4 = self.layers[4](f3)
        return (f0, f1, f2, f3, f4)
