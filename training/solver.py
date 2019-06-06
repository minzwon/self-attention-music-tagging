# coding: utf-8
import os
import time
import numpy as np
from sklearn import metrics
import datetime
import tqdm
import _pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import MTSA

try:
    import nsml
except ImportError:
    pass


class Solver(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader
        self.dataset = config.dataset
        if config.dataset == 'mtat':
            self.valid_list = np.load(os.path.join(config.data_path, 'data/valid_new.npy'))
            self.binary = np.load(os.path.join(config.data_path, 'data/binary.npy'))
        elif config.dataset == 'msd':
            self.valid_list = pkl.load(open(os.path.join(config.data_path, 'data/filtered_list_train.cP'), 'rb'))[201680:]
            self.binary = pkl.load(open(os.path.join(config.data_path, 'data/msd_id_to_tag_vector.cP'), 'rb'))
        self.data_type = config.data_type

        # Model hyper-parameters
        self.conv_channels = config.conv_channels
        self.attention_channels = config.attention_channels
        self.attention_layers = config.attention_layers
        self.attention_heads = config.attention_heads
        self.num_class = config.num_class
        self.input_length = config.input_length
        self.attention_length = config.attention_length
        self.attention_dropout = config.attention_dropout
        self.fc_dropout = config.fc_dropout

        # Training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.is_parallel = config.is_parallel
        self.use_tensorboard = config.use_tensorboard
        self.architecture = config.architecture

        # Path and step size
        self.data_path = config.data_path
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.model_save_epoch = config.model_save_epoch
        self.batch_size = config.batch_size

        # Cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.build_model()

        # use_nsml
        self.use_nsml = config.use_nsml

    def build_model(self):
        # model and optimizer
        model = MTSA(architecture=self.architecture,
                      conv_channels=self.conv_channels,
                      attention_channels=self.attention_channels,
                      attention_layers=self.attention_layers,
                      attention_heads=self.attention_heads,
                      attention_length=self.attention_length,
                      batch_size=self.batch_size,
                      attention_dropout=self.attention_dropout,
                      fc_dropout=self.fc_dropout,
                      is_cuda=self.is_cuda)

        self.model = model
        # cuda and parallel
        if self.is_parallel == 1:
            print('data parallel')
            self.model = torch.nn.DataParallel(self.model)
        if self.is_cuda == True:
            self.model.cuda()
        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def train(self):
        # Reconst loss
        reconst_loss = nn.BCELoss()

        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        best_roc_auc = 0
        drop_counter = 0
        for epoch in range(self.n_epochs):
            # train
            ctr = 0
            drop_counter += 1
            n_iters_per_epoch = len(self.data_loader)
            self.model.train()
            for x, y in self.data_loader:
                ctr += 1

                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                out = self.model(x)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                                datetime.timedelta(seconds=time.time()-start_t)))
                    if self.use_nsml:
                        step = epoch + ctr / n_iters_per_epoch
                        nsml.report(scope=locals(),
                                    train__loss=loss.item(),
                                    step=step)

            # validation
            roc_auc, pr_auc = self.get_validation_auc(num_chunks=16)
            if self.use_nsml:
                nsml.report(scope=locals(),
                            test__rocauc=roc_auc,
                            test__prauc=pr_auc,
                            step=epoch + 1)

            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save(self.model.state_dict(),
                        os.path.join(self.model_save_path, 'best_model.pth'))
            # change optimizer
            if current_optimizer == 'adam' and drop_counter == 60:
                self.load = os.path.join(self.model_save_path, 'best_model.pth')
                self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
                current_optimizer = 'sgd_1'
                drop_counter = 0
                print('sgd 1e-3')
            # first drop
            if current_optimizer == 'sgd_1' and drop_counter == 20:
                self.load = os.path.join(self.model_save_path, 'best_model.pth')
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.0001
                current_optimizer = 'sgd_2'
                drop_counter = 0
                print('sgd 1e-4')
            # second drop
            if current_optimizer == 'sgd_2' and drop_counter == 20:
                self.load = os.path.join(self.model_save_path, 'best_model.pth')
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.00001
                current_optimizer = 'sgd_3'
                print('sgd 1e-5')

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def get_auc(self, est_array, gt_array):
        est_array = np.array(est_array)
        gt_array = np.array(gt_array)

        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)
        return roc_aucs, pr_aucs

    def get_tensor(self, fn, num_chunks):
        if self.data_type == 'spec':
            # load audio
            if self.dataset == 'mtat':
                spec_path = os.path.join(self.data_path, self.data_type, fn.split('/')[1][:-3]) + 'npy'
            elif self.dataset == 'msd':
                spec_path = os.path.join(self.data_path, self.data_type, fn[2], fn[3], fn[4], fn +'.npy')
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
            if self.dataset == 'mtat':
                raw_path = os.path.join(self.data_path, self.data_type, fn.split('/')[1][:-3]) + 'npy'
            elif self.dataset == 'msd':
                raw_path = os.path.join(self.data_path, self.data_type, fn[2], fn[3], fn[4], fn + '.npy')
            raw = np.load(raw_path, mmap_mode='r')
            # split chunk
            length = len(raw)
            chunk_length = 65610
            hop = (length - chunk_length) // num_chunks
            x = torch.zeros(num_chunks, chunk_length)
            for i in range(num_chunks):
                x[i] = torch.Tensor(raw[i*hop:i*hop+chunk_length]).unsqueeze(0)
            return x

    def get_validation_auc(self, num_chunks=16):
        self.model.eval()
        est_array = []
        gt_array = []
        for _i, line in enumerate(self.valid_list):
            if self.dataset == 'mtat':
                ix, fn = line.split('\t')
            elif self.dataset == 'msd':
                fn = line
            try:
                # load and split
                x = self.get_tensor(fn, num_chunks)

                # forward
                x = self.to_var(x)
                out = self.model(x)
                out = out.detach().cpu()

                # estimate
                estimated = np.array(out).mean(axis=0)
                est_array.append(estimated)

                # ground truth
                if self.dataset == 'mtat':
                    ground_truth = self.binary[int(ix)]
                elif self.dataset == 'msd':
                    ground_truth = self.binary[fn].astype(int).reshape(50)
                gt_array.append(ground_truth)

                if _i % 200 == 0:
                    print("[%s] Valid Iter [%d/%d] " %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                _i, len(self.valid_list)))
            except FileNotFoundError:
                print(fn)
                continue

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc

