import torch.utils.data as data
from os import listdir
from os.path import isfile, join
import os
import json
import progressbar
from multiprocessing import Pool
import _pickle as pickle

class RagaDataset(data.Dataset):
    
    def _load_json(self, json_path):
        self.songs.append(json.load(open(json_path)))
        pickle.load(json_path.replace('json', 'spg').replace('metadata', 'spectr'))

    def __init__(self, data_root):
        self.songs = []
        print("Loading JSONS!")
        jsons = [join(data_root, 'metadata/', f) for f in listdir(join(data_root, 'metadata/')) if isfile(join(data_root, 'metadata/', f))]
        pool = Pool(os.cpu_count())
        pool.map(self._load_json, jsons)
        print("Done!")

a = RagaDataset('/home/sauhaarda/Dataset')
