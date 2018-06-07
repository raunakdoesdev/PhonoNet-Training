import torch.utils.data as data
from os import listdir
from os.path import isfile, join
import os
import json
import progressbar
from multiprocessing import Pool
import torch
import time, tqdm, random
from collections import deque

class RagaDataset(data.Dataset):
    
    def song_queue_push(self, json_path):
        spectr = torch.from_numpy(torch.load(json_path.replace('json', 'spg').replace('metadata', 'spectr')))
        json_file = json.load(open(json_path))
        return ((spectr, json_file,))

    def __init__(self, data_root, song_q_len):
        # Set json q length
        self.song_q_len = song_q_len

        # Load up all valid jsons
        self.json_q = [join(data_root, 'metadata/', f) for f in listdir(join(data_root, 'metadata/')) if isfile(join(data_root, 'metadata/', f)) and f.endswith('json')]

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
            for song in tqdm.tqdm(pool.imap_unordered(self.song_queue_push, self.json_q[0:num_grab]), total=num_grab):
                self.song_q.append(song)  # Add each loaded song to the queue

            if not len(self.song_q) == num_grab:  # pop items out
                self.json_q = self.json_q[num_grab:]
            else:  # list should be empty
                self.json_q = []
            print("Done!")

        return self.song_q.popleft()

a = RagaDataset('/home/sauhaarda/Dataset', 5)
