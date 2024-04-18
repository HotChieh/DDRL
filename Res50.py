import torch.nn as nn
import torch
from torchvision import models

# from misc.layer import Conv2d, FC

import torch.nn.functional as F
from utils import *

import pdb

# model_path = '../PyTorch_Pretrained/resnet50-19c8e357.pth'
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
class Res50(nn.Module):
    def __init__(self,  pretrained=True):
        super(Res50, self).__init__()

        self.de_pred_1 = nn.Sequential(Conv2d(256, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        self.de_pred_2 = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        self.de_pred_3 = nn.Sequential(Conv2d(256, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        self.Linear1 = nn.Sequential(nn.Linear(30*40, 128), nn.ReLU(inplace=True),
                                     )
        self.Linear2 = nn.Sequential(nn.Linear(128+25, 3), nn.ReLU(inplace=True),
                                     )
        # self.generator1 = nn.Sequential(Conv2d(1024, 512, 1, same_padding=True, NL='relu'),
        #                                nn.ConvTranspose2d(512, 512, 2, stride=2,padding=0), nn.ReLU(inplace=True),
        #                                Conv2d(512, 512, 1, same_padding=True, NL='relu')
        #                                )
        # self.generator2 = nn.Sequential(Conv2d(512, 256, 1, same_padding=True, NL='relu'),
        #                                nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0), nn.ReLU(inplace=True),
        #                                Conv2d(256, 256, 1, same_padding=True, NL='relu')
        #                                )
        # self.generator3 = nn.Sequential(Conv2d(256, 3, 1, same_padding=True, NL='relu'),
        #                                nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0), nn.ReLU(inplace=True),
        #                                Conv2d(3, 3, 1, same_padding=True, NL='relu')
                                    #    )
        initialize_weights(self.modules())

        res = models.resnet18(pretrained=pretrained)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        # self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        # self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

        


    def forward(self,img, state):
        _,_,H,W = img.shape
        
        x = self.frontend(img)
        x1 = self.layer1(x)
        

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # g1 = self.generator1(x3)
        # g2 = self.generator2(g1)
        # g3 = self.generator3(g2)

        # x1 = self.de_pred_1(x1)
        # x2 = self.de_pred_2(x2)
        x3 = self.de_pred_3(x3)
        b,c,h,w = x3.shape
        x_out = self.Linear1(x3.view(b, c*h*w))
        x_out = torch.cat((x_out, state),1)
        x_out = self.Linear2(x_out)
        # x2 = F.upsample(x2, size=(h, w))
        # x3 = F.upsample(x3, size=(h, w))
        # x_sum = x1+x2+x3
        # x = F.upsample(x_sum,size=(H, W))
        return x_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)   


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out        