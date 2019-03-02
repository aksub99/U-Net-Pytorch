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
        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv_norm_relu(channels_in, channels_out)
        )
    
    def forward(self, x):
        return self.conv(x)

class up(nn.Module):
    raise(NotImplementedError)

class right_out(nn.Module):
    raise(NotImplementedError)        