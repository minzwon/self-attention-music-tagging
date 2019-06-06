# coding: utf-8
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from .lee_conv import Conv1, Conv3, Conv7, InitConv


class LeeConvModule(nn.Module):
    def __init__(self, config):
        super(LeeConvModule, self).__init__()
        # initial convolution
        self.init_conv = InitConv(config.conv_channels)

        # stack convolution
        self.be = config.back_end
        c = config.conv_channels
        if config.back_end == 'won':
            channels = [c, c, c*2, c*2, c*2]
        elif config.back_end == 'lee':
            channels = [c, c, c*2, c*2, c*2, c*2, c*2, c*2, c*4, c*4]
        self.convs = nn.ModuleList([Conv3(channels[i], channels[i+1]) for i in range(len(channels)-1)])

        # 1x1 conv
        if config.back_end == 'won':
            self.conv1x1 = Conv7(channels[-1], config.attention_channels)
        elif config.back_end == 'lee':
            self.conv1x1 = Conv1(channels[-1], config.attention_channels)

    def forward(self, x):
        out = self.init_conv(x)
        for layer in self.convs:
            out = layer(out)
        out = self.conv1x1(out)
        if self.be == 'lee':
            out = out.squeeze(2)
        return out
