# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from .senet import senet154, se_resnext50_32x4d
from encoding.nn import SyncBatchNorm

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_classes, backbone='seresnext50', norm_layer=SyncBatchNorm):
        super(UNet, self).__init__()
        if backbone == 'seresnext50':
            self.backbone = se_resnext50_32x4d(norm_layer=SyncBatchNorm)
        self.up1 = up(3072, 768, norm_layer=norm_layer)
        self.up2 = up(1280, 320, norm_layer=norm_layer)
        self.up3 = up(576, 144, norm_layer=norm_layer)
        self.up4 = up(208, 64, norm_layer=norm_layer)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        imsize = x.size()[2:]
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.interpolate(x, imsize)
        return (torch.sigmoid(x), )
