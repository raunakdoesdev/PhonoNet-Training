from os import listdir
from os.path import isfile, join
import os
import json
import progressbar
from multiprocessing import Pool
import time, tqdm, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import logging
from collections import OrderedDict
from torch.utils.data import DataLoader


bptt = 5

torch.set_default_tensor_type('torch.FloatTensor')

def pad_tensor(vec, pad, dim):
    if pad - vec.size(dim) > 0:
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    else:
        return vec

class PadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):

        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        batch = map(lambda xy: (pad_tensor(xy[0], pad=max_len, dim=self.dim), xy[1]), batch)
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

class RagaDataset(object):
    
    def __init__(self, data_root):
        # Load up all valid jsons
        self.json_q = [join(data_root, 'metadata/', f) for f in listdir(join(data_root, 'metadata/')) if isfile(join(data_root, 'metadata/', f)) and f.endswith('json')]

        # Num songs
        self.num_songs = len(self.json_q) 

    def __getitem__(self, index):

        json_path = self.json_q[index]
        json_file = json.load(open(json_path))
        spectr_path = json_path.replace('json', 'npy').replace('metadata', 'npy_spectr')
        spectr = torch.from_numpy(np.load(spectr_path)).transpose(0, 1).unsqueeze(0).float()

        return spectr, json_file['myragaid']

    def __len__(self):
        return self.num_songs
        
def time_split(self, source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

if __name__ == '__main__':
    ds = RagaDataset('/home/sauhaarda/Dataset')
    train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=PadCollate(dim=1))

    for epoch in range(100):
        for x, y in train_loader:
            print(x.shape)
        print(y.shape)
