import numpy as np
import torch
import json
from os import listdir
from os.path import join, isfile


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

        print('Padding')
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        batch = map(
            lambda xy: (
                pad_tensor(
                    xy[0],
                    pad=max_len,
                    dim=self.dim),
                xy[1]),
            batch)
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=1)
        ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


class RagaDataset(object):

    def __init__(self, data_root):
        # Load up all valid jsons
        self.json_q = [join(data_root,
                            'metadata/',
                            f) for f in listdir(join(data_root,
                                                     'metadata/')) if isfile(join(data_root,
                                                                                  'metadata/',
                                                                                  f)) and f.endswith('json')]

        # Num songs
        self.num_songs = len(self.json_q)

    def __getitem__(self, index):
        json_path = self.json_q[index]
        json_file = json.load(open(json_path))
        spectr_path = json_path.replace(
            'json', 'npy').replace(
            'metadata', 'npy_spectr')
        spectr = torch.from_numpy(np.load(spectr_path)).transpose(0, 1).float()
        print('Loading Song')

        return spectr, json_file['myragaid']

    def __len__(self):
        return self.num_songs
