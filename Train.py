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


torch.set_default_tensor_type('torch.FloatTensor')

class RagaDataset(object):
    
    def song_queue_push(self, json_path):
        spectr = torch.from_numpy(np.load(json_path.replace('json', 'npy'). \
	    replace('metadata', 'npy_spectr')))

        json_file = json.load(open(json_path))
        return ((spectr, json_file,))

    def __init__(self, data_root, song_q_len):
        # Set json q length
        self.song_q_len = song_q_len

        # Load up all valid jsons
        self.json_q = [join(data_root, 'metadata/', f) for f in listdir(join(data_root, \
		'metadata/')) if isfile(join(data_root, 'metadata/', f)) and f.endswith('json')]

        # Num songs
        self.num_songs = len(self.json_q) 

        # Shuffle JSON Queue
        random.shuffle(self.json_q)
        
        # Initialize empty song q
        self.song_q = deque([])

    def __getitem__(self, index):
        if len(self.song_q) == 0:  # Refill the song queue
            num_grab = min(self.song_q_len, len(self.json_q))  # num elements to pop

            print("Loading more songs!")

            # Multithreaded loading of songs detailed in json queue
            pool = Pool(os.cpu_count())
            for song in tqdm.tqdm(pool.imap_unordered(
		self.song_queue_push, self.json_q[0:num_grab]), total=num_grab):

                self.song_q.append(song)  # Add each loaded song to the queue

            if not len(self.song_q) == num_grab:  # pop items out
                self.json_q = self.json_q[num_grab:]
            else:  # list should be empty
                self.json_q = []
            print("Done!")

        return self.song_q.popleft()

    def __len__(self):
        return self.num_songs

class RagaDetector(nn.Module):
    def __init__(self, num_outputs):
        super(RagaDetector, self).__init__()

        self.num_outputs = num_outputs

        self.encoder = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(1)), 

            ('conv1', nn.Conv2d(1, 64, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(64)), 
            ('elu1', nn.ELU()),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('drop1', nn.Dropout(p=0.1)),

            ('conv2', nn.Conv2d(64, 128, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(128)), 
            ('elu2', nn.ELU()),
            ('pool2', nn.MaxPool2d(3, 3)),
            ('drop2', nn.Dropout(p=0.1)),

            ('conv3', nn.Conv2d(128, 128, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(128)), 
            ('elu3', nn.ELU()),
            ('pool3', nn.MaxPool2d(4, 4)),
            ('drop3', nn.Dropout(p=0.1)),

            ('conv4', nn.Conv2d(128, 128, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(128)), 
            ('elu4', nn.ELU()),
            ('pool4', nn.MaxPool2d(4, 4)),
            ('drop4', nn.Dropout(p=0.1))
        ]))

        self.lstm_1 = torch.nn.LSTMCell(640, 100)
        self.lstm_2 = torch.nn.LSTMCell(100, num_outputs)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.encoder(x)
        x = torch.split(x, 5, dim=3)[:-1]

        hx_1 = torch.randn(batch_size, 100)
        cx_1 = torch.randn(batch_size, 100)

        hx_2 = torch.randn(batch_size, self.num_outputs)
        cx_2 = torch.randn(batch_size, self.num_outputs)

        output = []
        for i in range(len(x)):
            input = x[i].reshape((batch_size, -1))
            hx_1, cx_1 = self.lstm_1(input, (hx_1, cx_1))
            hx_2, cx_2 = self.lstm_2(hx_1, (hx_2, cx_2))
            output.append(hx_2)

        return x

if __name__ == '__main__':
    dataset = RagaDataset('/home/sauhaarda/Dataset', 3)
    net = RagaDetector(72)
    # data = dataset[0][0].unsqueeze(0).unsqueeze(0).float()
    # print(data.shape)
    # print(net(data)[0].shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for x, y in dataset:

            label = y['myragaid']
            labels = torch.zeros(1, 72)
            labels[0][i] = 1

            optimizer.zero_grad()
            outputs = net(x.unsqueeze(0).unsqueeze(0).float())[-1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            break
        break
