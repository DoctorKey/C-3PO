# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

from model.unet import Unet

class SiamUnet_conc(Unet):
    """SiamUnet_conc segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(SiamUnet_conc, self).__init__(input_nbr, label_nbr)
        self.conv43d = nn.ConvTranspose2d(384, 128, kernel_size=3, padding=1)
        self.conv33d = nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1)
        self.conv22d = nn.ConvTranspose2d(96, 32, kernel_size=3, padding=1)
        self.conv12d = nn.ConvTranspose2d(48, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x1, x2 = torch.split(x,3,1)

        """Forward method."""
        f1 = super(SiamUnet_conc, self).extract_feature(x1)
        x12_1, x22_1, x33_1, x43_1, x4p = f1

        f2 = super(SiamUnet_conc, self).extract_feature(x2)
        x12_2, x22_2, x33_2, x43_2, x4p = f2

        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2), 1)
        x41d = self.stage_4d(x4d)

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2), 1)
        x31d = self.stage_3d(x3d)

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2), 1)
        x11d = self.stage_1d(x1d)
        return x11d
        #return self.sm(x11d)

def FC_Siam_conc(args):
    return SiamUnet_conc(input_nbr=3, label_nbr=2)