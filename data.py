import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from os import listdir
from os.path import join, isfile
import glob, random
import multiprocessing

class RagaDataset(object):

    def __init__(self, data_root, json_q=None, max_len=3000):
        if json_q is None:
            self.json_q = glob.glob(join(data_root, 'meta_chunks') + '/**/*.json', recursive=True)
        else:
            self.json_q = json_q

        self.max_len = max_len

        # Num songs
        self.num_songs = len(self.json_q)

    def __getitem__(self, index):
        json_path = self.json_q[index]
        json_file = json.load(open(json_path))

        spectr_path = json_path.replace(
            'json', 'npy').replace(
            'meta_chunks', 'chroma_chunks')
        spectr = torch.from_numpy(np.load(spectr_path)).float()

        # print(spectr.size())

        if not spectr.size()[1] == self.max_len:
            padding = torch.zeros(spectr.size()[0], self.max_len - spectr.size()[1])
            spectr = torch.cat((spectr, padding), 1)


        return spectr.unsqueeze(0), json_file['myragaid']

    def __len__(self):
        return self.num_songs

def get_dataloaders(data_root='/home/sauhaarda/Dataset', split=0.9, seed=141, batch_size=10):
    # Load queue of all the songs 
    json_q = glob.glob(join(data_root, 'meta_chunks') + '/**/*.json', recursive=True)
    random.Random(seed).shuffle(json_q)

    # Create train/val split
    train_len = int(len(json_q) * split)
    print(train_len)
    val_len = len(json_q) - train_len

    train_q = json_q[:train_len]
    val_q = json_q[-val_len:]

    # Create Datset objects
    td = RagaDataset(data_root, train_q)
    vd = RagaDataset(data_root, val_q)

    train_loader = DataLoader(
        td,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count(),
        shuffle=True)
    val_loader = DataLoader(
        vd,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count(),
        shuffle=False)
    return train_loader, val_loader
