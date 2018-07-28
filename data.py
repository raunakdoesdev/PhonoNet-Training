import numpy as np
import torch
import json
from os import listdir
from os.path import join, isfile
import glob


class RagaDataset(object):

    def __init__(self, data_root):
        # Load up all valid jsons
        self.json_q = glob.glob(join(data_root, 'meta_chunks') + '/**/*.json', recursive=True)

        # Num songs
        self.num_songs = len(self.json_q)

    def __getitem__(self, index):
        json_path = self.json_q[index]
        json_file = json.load(open(json_path))

        spectr_path = json_path.replace(
            'json', 'npy').replace(
            'meta_chunks', 'spectr_chunks')
        spectr = torch.from_numpy(np.load(spectr_path)).float()

        if not spectr.size()[1] == 6000:
            padding = torch.zeros(spectr.size()[0], 6000 - spectr.size()[1])
            spectr = torch.cat((spectr, padding), 1)

        return spectr.unsqueeze(0), json_file['myragaid']

    def __len__(self):
        return self.num_songs

# d = RagaDataset()
