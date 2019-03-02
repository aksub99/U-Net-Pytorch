import torch
import torch.nn as nn
import torch.nn.functional as F

# SAME padding will be used in this implementation unlike original paper where padding = NONE
class UNet(nn.Module):
    def __init__(self, no_channels, no_classes):
        super(UNet, self).__init__()
        # right_in used only to increase channels from initial image(3/1) to 64
        self.right_in = inc(no_channels, 64) 
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
        # right_out used only to reduce number of channels from 64 to number of classes in output layer 
        self.right_out = red(64, no_classes)
    
    def forward(self, x):
        raise(NotImplementedError)
