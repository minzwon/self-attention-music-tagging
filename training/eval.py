# coding: utf-8
import os
import time
import numpy as np
import datetime
import tqdm
import fire
import argparse
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import MTSA


class Predict(object):
    def __init__(self, config):
        # Model hyper-parameters
        self.architecture = config.architecture
        self.conv_channels = config.conv_channels
        self.attention_channels = config.attention_channels
        self.attention_layers = config.attention_layers
        self.attention_heads = config.attention_heads
        self.attention_length = config.attention_length
        self.num_class = config.num_class
        self.dataset = config.dataset
        self.data_type = config.data_type
        self.input_length = config.input_length
        self.batch_size = config.batch_size

        # Path and step size
        self.audio_path = config.data_path

        # Cuda
        self.is_cuda = torch.cuda.is_available()
        self.is_parallel = config.is_parallel

        # Build model
        self.build_model()

        # Start with trained model
        self.load(config.model_path)

    def build_model(self):
        # model and optimizer
        model = MTSA(architecture=self.architecture,
                      conv_channels=self.conv_channels,
                      attention_channels=self.attention_channels,
                      attention_layers=self.attention_layers,
                      attention_heads=self.attention_heads,
                      attention_length=self.attention_length,
                      num_class=self.num_class,
                      batch_size=self.batch_size,
                      is_cuda=self.is_cuda)
        self.model = model
        # data parallel
        if self.is_parallel == 1:
            self.model = torch.nn.DataParallel(self.model)
        # cuda
        if self.is_cuda == True:
            self.model.cuda()

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def detach(self, x):
        return Variable(x.data)

    def get_tensor(self, fn, num_chunks):
        if self.data_type == 'spec':
            # load audio
            spec_path = os.path.join(self.audio_path, self.data_type, fn.split('/')[1][:-3]) + 'npy'
            spec = np.load(spec_path, mmap_mode='r')
            # split chunk
            length = spec.shape[1]
            chunk_length = self.input_length
            hop = (length - chunk_length) // num_chunks
            x = torch.zeros(num_chunks, 96, chunk_length)
            for i in range(num_chunks):
                x[i] = torch.Tensor(spec[:, i*hop:i*hop+chunk_length]).unsqueeze(0)
            return x
        elif self.data_type == 'raw':
            # load audio
            raw_path = os.path.join(self.audio_path, self.data_type, fn.split('/')[1][:-3]) + 'npy'
            raw = np.load(raw_path, mmap_mode='r')
            # split chunk
            length = len(raw)
            chunk_length = self.input_length
            hop = (length - chunk_length) // num_chunks
            x = torch.zeros(num_chunks, chunk_length)
            for i in range(num_chunks):
                x[i] = torch.Tensor(raw[i*hop:i*hop+chunk_length]).unsqueeze(0)
            return x

    def forward(self, x):
        x = self.to_var(x)
        x = self.model(x)
        x = self.detach(x)
        return x.cpu()

    def get_auc(self, est_array, gt_array):
        est_array = np.array(est_array)
        gt_array = np.array(gt_array)

        roc_auc = []
        pr_auc = []
        for _tag in range(50):
            roc_auc.append(metrics.roc_auc_score(gt_array[:, _tag], est_array[:, _tag]))
            pr_auc.append(metrics.average_precision_score(gt_array[:, _tag], est_array[:, _tag]))
        print('roc_auc: %.4f' % np.mean(roc_auc))
        print('pr_auc: %.4f' % np.mean(pr_auc))

    def evaluate(self, num_chunks=16):
        self.evaluate_auto_tagging(num_chunks)

    def evaluate_auto_tagging(self, num_chunks=16):
        self.model.eval()
        if self.dataset == 'mtat':
            filelist = np.load(os.path.join(self.audio_path, 'data/test_new.npy'))
        binary = np.load(os.path.join(self.audio_path, 'data/binary.npy'))

        est_array = []
        gt_array = []
        for line in tqdm.tqdm(filelist):
            ix, fn = line.split('\t')
            # load and split
            x = self.get_tensor(fn, num_chunks)

            # forward
            prd = self.forward(x)

            # estimated
            estimated = np.array(prd).mean(axis=0)
            est_array.append(estimated)

            # ground truth
            ground_truth = binary[int(ix)]
            gt_array.append(ground_truth)

        # get roc_auc and pr_auc
        self.get_auc(est_array, gt_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model parameters
    parser.add_argument('--input_length', type=int, default=256)
    parser.add_argument('--attention_length', type=int, default=271)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--conv_channels', type=int, default=16)
    parser.add_argument('--attention_channels', type=int, default=512)
    parser.add_argument('--attention_layers', type=int, default=2)
    parser.add_argument('--attention_heads', type=int, default=8)

    parser.add_argument('--num_class', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='mtat')
    parser.add_argument('--data_type', type=str, default='spec')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--is_parallel', type=int, default=1)

    parser.add_argument('--architecture', type=str)

    config = parser.parse_args()

    p = Predict(config)
    p.evaluate()






