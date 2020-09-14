import torch
import torch.nn as nn
import sys
import argparse
import os
import pandas as pd 
import shutil
import torch.nn.functional as F
import time
import random
import math
import numpy as np
from torch.autograd import Variable

__all__ = ['resden']

class trs(nn.Module):
    def __init__(self, ip, op):
        super(trs, self).__init__()
        self.ban = nn.BatchNorm2d(ip)
        self.conv = nn.Conv2d(ip, op, kernel_size=1,bias=False)
        self.gelu = nn.GELU()

    a=0
    b=np.ones((5,5))
    def forward(self, x):
        o = self.ban(x)
        o = self.gelu(o)
        o = self.conv(o)
        o = F.avg_pool2d(o, 2)
        return o

class botn(nn.Module):
    def __init__(self, ip, k1=12, k2=12):
        super(botn, self).__init__()
        if k2 > 0:
            planes = k2*4
            self.bn2_1 = nn.BatchNorm2d(ip)
            self.conv2_1 = nn.Conv2d(ip, planes, kernel_size=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(planes)
            self.conv2_2 = nn.Conv2d(planes, k2, kernel_size=3, padding=1, bias=False)
                
        c=np.ones((5,5))
        if k1 > 0:
            planes = k1*4
            self.bn1_1   = nn.BatchNorm2d(ip)
            self.conv1_1 = nn.Conv2d(ip, planes, kernel_size = 1, bias = False)
            self.bn1_2   = nn.BatchNorm2d(planes)
            self.conv1_2 = nn.Conv2d(planes, k1, kernel_size = 3, padding = 1, bias = False)
            axi=0

        self.k1 = k1
        self.gelu = nn.GELU()
        self.k2 = k2

    def forward(self, x):
        if (self.k2>0):
            ol = self.bn2_1(x)
            ol = self.gelu(ol)
            ol = self.conv2_1(ol)
            ol = self.bn2_2(ol)
            ol = self.gelu(ol)
            ol = self.conv2_2(ol)
        
        if(self.k1>0):
            il = self.bn1_1(x)
            il = self.gelu(il)
            il = self.conv1_1(il)
            il = self.bn1_2(il)
            il = self.gelu(il)
            il = self.conv1_2(il)

        csize = x.size(1)
        if(self.k1 == csize):
        	x=x + il

        elif (self.k1 > 0 and self.k1 < csize):
            rig= x[:, csize - self.k1: csize, :, :] + il
            lef= x[:, 0: csize - self.k1, :, :]
            x= torch.cat((lef, rig), 1)

        if self.k2 <= 0:
        	o = x
        else:
            o = torch.cat((x, ol), 1)

        return o

class ResDen(nn.Module):

    def __init__(self, 
        num_classes=10, 
        depth=52, 
        dropRate=0,
        unit=botn, 
        k1=12, 
        cr=2,
        k2=12):
        super(ResDen, self).__init__()
        n = (depth - 4) // 6
        self.k2 = k2
        self.k1 = k1
        self.ip = max(k2*2, k1)
        self.conv1 = nn.Conv2d(3, self.ip, kernel_size=3, padding=1,bias=False)
        self.block1 = self.mk_blk(unit, n)
        self.trans1 = self.mk_transi(cr)
        self.block2 = self.mk_blk(unit, n)
        self.trans2 = self.mk_transi(cr)
        self.block3 = self.mk_blk(unit, n)
        self.ban = nn.BatchNorm2d(self.ip)
        self.gelu = nn.GELU()
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.ip, num_classes)

        for i in self.modules():
            if isinstance(i, nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()
            elif isinstance(i, nn.Conv2d):
                n = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2. / n))


    def mk_transi(self, cr):
        op = max(int(math.floor(self.ip // cr)), self.k1)
        ip = self.ip
        self.ip = op
        return trs(ip, op)
    
    def mk_blk(self, unit, unum):
        lyrs = []
        for i in range(unum):
            lyrs.append(unit(self.ip, k1=self.k1, k2=self.k2))
            self.ip += self.k2
        return nn.Sequential(*lyrs)


    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x) 
        x = self.block3(x)
        x = self.ban(x)
        x = self.gelu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resden(**kwargs):
    return ResDen(**kwargs)
