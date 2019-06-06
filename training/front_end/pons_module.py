# coding: utf-8
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from .pons_conv import VerticalConv, HorizontalConv


class PonsConvModule(nn.Module):
    def __init__(self, config):
        super(PonsConvModule, self).__init__()
        self.input_bn = nn.BatchNorm2d(1)
        m1 = VerticalConv(1, config.conv_channels, (int(0.9*96), 7))
        m2 = VerticalConv(1, config.conv_channels*2, (int(0.9*96), 3))
        m3 = VerticalConv(1, config.conv_channels*4, (int(0.9*96), 1))
        m4 = VerticalConv(1, config.conv_channels, (int(0.4*96), 7))
        m5 = VerticalConv(1, config.conv_channels*2, (int(0.4*96), 3))
        m6 = VerticalConv(1, config.conv_channels*4, (int(0.4*96), 1))
        m7 = HorizontalConv(1, config.conv_channels, 165)
        m8 = HorizontalConv(1, config.conv_channels*2, 129)
        m9 = HorizontalConv(1, config.conv_channels*4, 65)
        m10 = HorizontalConv(1, config.conv_channels*8, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])
        self.be = config.back_end
        if config.back_end == 'won':
            self.conv1x1 = nn.Conv1d(config.conv_channels*29, config.attention_channels, 1)
            self.bn = nn.BatchNorm1d(config.attention_channels)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_bn(x)
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)
        if self.be == 'won':
            out = self.relu(self.bn(self.conv1x1(out)))

        return out
