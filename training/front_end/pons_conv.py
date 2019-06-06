# coding: utf-8
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class VerticalConv(nn.Module):
    def __init__(self, input_channels, output_channels, filter_shape):
        super(VerticalConv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, filter_shape,
                              padding=(0, filter_shape[1]//2))
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(self.relu(self.bn(self.conv(x))))
        out = out.squeeze(2)
        return out

class HorizontalConv(nn.Module):
    def __init__(self, input_channels, output_channels, filter_length):
        super(HorizontalConv, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, filter_length, padding=filter_length//2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out
