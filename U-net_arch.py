from .U-net_modules_defn import *
import torch.nn as nn
import torch.nn.functional as F


# SAME padding will be used in this implementation unlike //
# original paper where padding = NONE
class UNet(nn.Module):
    def __init__(self, no_channels, no_classes):
        super(UNet, self).__init__()
        # right_in used only to increase channels from initial image(3/1) to 64
        self.right_in = increase(no_channels, 64)
        # down will include (conv, BN, ReLU)*2 and maxpool(2*2 stride 2)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        # up will include upconv, concatenation, (conv, BN, ReLU)*2
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        # right_out used only to reduce number of channels //
        # from 64 to number of classes in output layer
        self.right_out = decrease(64, no_classes)

    def forward(self, x):
        # function to define sequence of operations to be performed
        x1 = self.right_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5)
        x7 = self.up2(x6)
        x8 = self.up3(x7)
        x9 = self.up4(x8)
        x10 = self.right_out(x9)
        return F.sigmoid(x10)
