# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config

from front_end.pons_module import PonsConvModule
from front_end.lee_module import LeeConvModule
from back_end.won_module import AttentionModule
from back_end.pons_module import PonsBackendModule

class MTSA(nn.Module):
    def __init__(self,
                architecture='pons_won',
                conv_channels=16,
                attention_channels=512,
                attention_layers=2,
                attention_heads=8,
                attention_length=257,
                num_class=50,
                batch_size=16,
                attention_dropout=0.1,
                fc_dropout=0.5,
                is_cuda=True):
        super(MTSA, self).__init__()
        self.is_cuda = is_cuda
        self.be = architecture.split('_')[1]

        # Configuration
        config = Config(back_end=self.be,
                        conv_channels=conv_channels,
                        attention_channels=attention_channels,
                        attention_layers=attention_layers,
                        attention_heads=attention_heads,
                        attention_length=attention_length,
                        num_class=num_class,
                        batch_size=batch_size,
                        attention_dropout=attention_dropout,
                        fc_dropout=fc_dropout,
                        is_cuda=is_cuda)

        # Front-end
        self.front_end = self.get_front_end(config, architecture)

        # Back-end
        self.back_end = self.get_back_end(config, architecture)

        # Output layer
        self.dropout = nn.Dropout(fc_dropout)
        if self.be == 'pons':
            self.classifier = nn.Linear(500, num_class)
        else:
            self.classifier = nn.Linear(attention_channels, num_class)

        # Sigmoid
        self.activation = nn.Sigmoid()

        # [CLS] vector
        if self.be == 'won':
            self.vec_cls = self.get_cls(config.batch_size, config.attention_channels, config.is_cuda)

    def get_front_end(self, config, architecture):
        if architecture.split('_')[0] == 'pons':
            return PonsConvModule(config)
        elif architecture.split('_')[0] == 'lee':
            return LeeConvModule(config)

    def get_back_end(self, config, architecture):
        if architecture.split('_')[1] == 'won':
            return AttentionModule(config)
        elif architecture.split('_')[1] == 'pons':
            return PonsBackendModule(config)
        elif architecture.split('_')[1] == 'lee':
            return None

    def get_cls(self, batch, channel, is_cuda):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        #if is_cuda == True:
        #    single_cls = single_cls.cuda()
        vec_cls = torch.cat([single_cls for _ in range(32)], dim=0) # maximum batch size is 32 now
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Front-end
        out = self.front_end(x)

        # Permute
        if self.be == 'won':
            out = out.permute(0, 2, 1)
            out = self.append_cls(out)

        # Back-end
        if self.be == 'won' or self.be == 'pons':
            out = self.back_end(out)

        # Dense
        out = self.dropout(out)
        logits = self.activation(self.classifier(out))

        return logits
