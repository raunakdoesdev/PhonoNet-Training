import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from random import shuffle
from os import listdir
from os.path import join, isfile
import glob, random
import multiprocessing
import pandas as pd

class RagaDataset(object):

    def __init__(self, df, max_len=1500, transform=None):
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

        if self.transform:
            return self.transform(chroma.unsqueeze(0), chunk['raga_id'], chunk['tonic'], chunk['song_id'])

        return chroma.unsqueeze(0), chunk['raga_id'], chunk['tonic'], chunk['song_id']

    def __len__(self):
        return self.df.shape[0]

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_dataloaders(song_split_num, data_path='/home/sauhaarda/Dataset/longdataset.pkl', transform=None, batch_size=10, valpart=0):
    df = pd.read_pickle(data_path)
    songs = torch.load('12fold.pkl')
    songs = [item for sublist in songs for item in sublist]
    val_songs = [songs.pop(valpart),]

    # Create Datset objects
    td = RagaDataset(df.loc[df['song_id'].isin(songs)], transform=transform)
    vd = RagaDataset(df.loc[df['song_id'].isin(val_songs)], transform=None)

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

    return train_loader, val_loader, val_songs
