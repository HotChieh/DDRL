from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
# from misc.layer import Conv2d, FC
from torchvision import models
from utils import *
import torchvision.ops

# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])

        self.b1 = Conv2d(512, 256, kernel_size=3, same_padding=True, NL='relu')
        self.b2 = Conv2d(256, 128, kernel_size=3, same_padding=True, NL='relu')
        self.b3 = Conv2d(128, 2, kernel_size=3, same_padding=True, NL='relu')
        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))



    def forward(self, x):
        _,_,H,W = x.shape
        x = self.features4(x)       
        dm = self.de_pred(x)

        dm = F.upsample(dm,size=(H, W))
        bbox_out = self.b3(self.b2(self.b1(x)))
        bbox_out = F.upsample(bbox_out,size=(H, W))
        return dm, bbox_out