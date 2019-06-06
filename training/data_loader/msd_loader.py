# coding: utf-8
import os
import numpy as np
from torch.utils import data
import _pickle as pkl

class AudioFolder(data.Dataset):
    def __init__(self, root, trval='TRAIN', dataset=None, data_type=None):
        self.trval = trval
        self.root = root
        self.dataset = dataset
        self.data_type = data_type
        if data_type == 'spec':
            self.input_length = 256
        elif data_type == 'raw':
            self.input_length = 65610
        self.get_songlist()
        self.tags = pkl.load(open(os.path.join(self.root, 'data/msd_id_to_tag_vector.cP'), 'rb'))

    def __getitem__(self, index):
        if self.data_type == 'spec':
            spec = []
            while len(spec) == 0:
                try:
                    spec, tag_binary = self.get_spec(index)
                except FileNotFoundError:
                    index = int(np.random.random(1) * len(self.fl))
                    spec = []
            return spec.astype('float32'), tag_binary.astype('float32')
        elif self.data_type == 'raw':
            raw = []
            while len(raw) == 0:
                try:
                    raw, tag_binary = self.get_raw(index)
                except FileNotFoundError:
                    index = int(np.random.random(1) * len(self.fl))
                    raw = []
            return raw.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        train = pkl.load(open(os.path.join(self.root, 'data/filtered_list_train.cP'), 'rb'))
        if self.trval == 'TRAIN':
            self.fl = train[:201680]
        elif self.trval == 'VALID':
            self.fl = train[201680:]
        elif self.trval == 'TEST':
            self.fl = pkl.load(open(os.path.join(self.root, 'data/filtered_list_test.cP'), 'rb'))

    def get_spec(self, index):
        fn = self.fl[index]
        spec_path = os.path.join(self.root, 'spec', fn[2], fn[3], fn[4], fn + '.npy')
        spec = np.load(spec_path, mmap_mode='r')
        if self.trval == 'TRAIN' or self.trval == 'VALID':
            random_idx = int(np.floor(np.random.random(1) * ((29*16000/256)-self.input_length)))
            spec = np.array(spec[:, random_idx:random_idx+self.input_length])
        tag_binary = self.tags[fn].astype(int).reshape(50)
        return spec, tag_binary

    def get_raw(self, index):
        fn = self.fl[index]
        raw_path = os.path.join(self.root, 'raw', fn[2], fn[3], fn[4], fn + '.npy')
        raw = np.load(raw_path, mmap_mode='r')
        if self.trval == 'TRAIN' or self.trval == 'VALID':
            random_idx = int(np.floor(np.random.random(1) * ((29*16000)-self.input_length)))
            raw = np.array(raw[random_idx:random_idx+self.input_length])
        tag_binary = self.tags[fn].astype(int).reshape(50)
        return raw, tag_binary

    def __len__(self):
        return len(self.fl)


def get_audio_loader(root, batch_size, trval='TRAIN', num_workers=0, dataset=None, data_type=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, trval=trval, dataset=dataset, data_type=data_type),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

