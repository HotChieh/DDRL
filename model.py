# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: DRL_Lane Pytorch 实现
# model.py

import torch.nn as nn
import pdb
import torch


class DRL_LANE(nn.Module):
    def __init__(self, cfg):
        super(DRL_LANE,self).__init__()
        # 使用默认的strid和padding
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(256, 256, kernel_size=3),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(256, 256, kernel_size=3),
            # nn.ReLU(),
            )
        self.fc1 = nn.Sequential(
            nn.Linear(11264, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(549, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 3))
        # self.dropout = nn.Dropout(p=0.4)
    def forward(self, img, state):
        img = self.encoder(img)
        img = img.view(img.shape[0],-1)
        img = self.fc1(img)
        img = torch.cat((img, state),1)
        img = self.fc2(img)
        # img = self.dropout(img)
        img = self.fc3(img)
        return img


def getModel(cfg):
    net = DRL_LANE(cfg)
    return net
