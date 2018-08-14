import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from os import listdir
from os.path import join, isfile
import glob, random
import multiprocessing
import pandas as pd

class RagaDataset(object):

    def __init__(self, df, max_len=3000, transform=None):
        if isinstance(df, pd.DataFrame):
            self.df = df
        elif isinstance(df, str):
            self.df = pd.read_pickle(df)
        else:
            raise TypeError("Input must be either a string or dataframe.")

        self.max_len = max_len
        self.transform = transform

    def __getitem__(self, index):
        chunk = self.df.iloc[index]
        chroma = torch.from_numpy(chunk['chroma']).float()
        if not chroma.size()[1] == self.max_len:
            padding = torch.zeros(chroma.size()[0], self.max_len - chroma.size()[1])
            chroma = torch.cat((chroma, padding), 1)

        if self.transform:
            return self.transform(chroma.unsqueeze(0), chunk['raga_id'], chunk['tonic'], chunk['song_id'])

        return chroma.unsqueeze(0), chunk['raga_id'], chunk['tonic'], chunk['song_id']

    def __len__(self):
        return self.df.shape[0]

def get_dataloaders(data_path='/home/sauhaarda/Dataset/dataset.pkl', split=0.98, seed=141, batch_size=10, transform=None):
    df = pd.read_pickle(data_path)

    songs = df.song_id.unique()
    random.Random(seed).shuffle(songs) # shuffle songs with random seed

    # Create train/val split
    train_len = int(len(songs) * split)
    print(train_len)
    val_len = len(songs) - train_len

    train_q = songs[:train_len]
    val_q = songs[-val_len:]

    # Create Datset objects
    td = RagaDataset(df.loc[df['song_id'].isin(train_q)], transform=transform)
    vd = RagaDataset(df.loc[df['song_id'].isin(val_q)], transform=None)

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

    return train_loader, val_loader, val_q
