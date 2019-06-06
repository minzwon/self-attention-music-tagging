# coding: utf-8
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class Conv1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, 1)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Conv7(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv7, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, 7, padding=3)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Conv3(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv3, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, 3, padding=1)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class InitConv(nn.Module):
    def __init__(self, output_channels):
        super(InitConv, self).__init__()
        self.conv = nn.Conv1d(1, output_channels, 3, stride=3, padding=1)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out
