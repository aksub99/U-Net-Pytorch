import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_norm_relu(nn.Module):
    # This class defines a double conv norm relu block
    def __init__(self, channels_in, channels_out):
        super(conv_norm_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class right_in(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(right_in, self).__init__()
        self.conv = conv_norm_relu(channels_in, channels_out)
    
    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv_norm_relu(channels_in, channels_out)
        )
    
    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = conv_norm_relu(channels_in, channels_out)
    
    # Assuming the input is (C, H, W)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffH = x2.size()[0] - x1.size()[0]
        diffW = x2.size()[1] - x2.size()[1]
        x1 = F.pad(x1,(diffW // 2, diffW - diffW//2, diffH // 2, diffH - diffH//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class right_out(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(right_out, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x):
        return self.conv(x)

