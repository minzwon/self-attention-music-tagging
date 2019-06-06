# coding: utf-8
import numpy as np
import torch
import torch.nn as nn

class PonsBackendModule(nn.Module):
    def __init__(self, config):
        super(PonsBackendModule, self).__init__()

        conv_channels = config.conv_channels
        intermediate_channels = config.attention_channels
        num_class = config.num_class

        # Jordi backend
        self.conv1 = nn.Conv1d(conv_channels*29, intermediate_channels, 7, padding=3)
        self.conv2 = nn.Conv1d(intermediate_channels, intermediate_channels, 7, padding=3)
        self.conv3 = nn.Conv1d(intermediate_channels, intermediate_channels, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.bn3 = nn.BatchNorm1d(intermediate_channels)

        self.dense = nn.Linear(intermediate_channels*2, 500)
        self.bn4 = nn.BatchNorm1d(500)
        self.relu = nn.ReLU()

    def forward(self, x):
        length = x.size(2)

        # Jordi
        res1 = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(res1))) + res1
        out = nn.MaxPool1d(2)(out)

        res2 = out
        out = self.relu(self.bn3(self.conv3(out))) + res2

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn4(self.dense(out)))

        return out
