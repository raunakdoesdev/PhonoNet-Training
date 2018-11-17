import numpy as np
import os
import pandas as pd
import torch
from models import RagaDetector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.nn.DataParallel(RagaDetector(0.5, 256).to(device))
# save = torch.load('saves/full_net_acc_check4_epoch_329_acc_0.6865671641791045.model')
#save = torch.load('saves/full_net_acc_check4_epoch_239_loss_1.4376170635223389.model')
save = torch.load('saves/0/full_net_acc_check4_epoch_86_acc_0.49242424242424243.model')
# save = torch.load('saves/0/full_net_acc_check4_epoch_181_loss_2.8628162145614624.model')

model.load_state_dict(save['net'])

criterion = torch.nn.CrossEntropyLoss()

df = pd.read_pickle('/home/sauhaarda/Dataset/song2raga.pkl')
song2raga = dict(zip(df.song_id, df.raga_id))

# Chunks function
def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n].swapaxes(0, 1)

class RagaDataset(object):
    def __init__(self, root, song_list=[]):
        self.song_list = []
        for song in song_list:
            a = np.load(os.path.join(root, song + '.npy'))
            self.song_list.append((a, song2raga[song],))

    def __getitem__(self, index):
        return self.song_list[index]

    def __len__(self):
        return len(self.song_list)

dat = RagaDataset('/home/sauhaarda/Dataset/chroma_real/', save['val_songs'])

model.eval()
segnumtot = [0] * 100
segnumcount = [0.] * 100

net = 0;
net_count = 0;

correct = 0.
tot = 0.
for song, raga in dat:
    song = song.swapaxes(0, 1) 
    chunks = list(chunk(song, 3000))
    net_prediction = None

    print("New song!")
    for segnum, segment in enumerate(chunks):
        if segment.shape[1] != 3000:
            padding = torch.zeros(segment.shape[0], 3000 - segment.shape[1]).double()
            segment = torch.cat((torch.from_numpy(segment), padding), 1).unsqueeze(0).unsqueeze(0);
        else:
            segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(0);
        out = model(segment.float().to(device))

        scaler = torch.max(out).item() ** 2
        if net_prediction is None:
            net_prediction = out * scaler
        else:
            net_prediction += out * scaler
            

        label = torch.LongTensor([raga]).to(device);
        raga_loss = criterion(out[:, :30], label)
        # print(raga_loss.item()) 
        segnumtot[segnum] += raga_loss.item()
        segnumcount[segnum] += 1

        _, predicted = torch.max(out[:, :30].data, 1)
        isCorrect = ((predicted == label).sum().item())
        net += isCorrect;
        net_count += 1;
        print(isCorrect)

    _, predicted = torch.max(net_prediction[:, :30].data, 1)
    isCorrect = ((predicted == label).sum().item())
    correct += isCorrect 
    tot += 1 
    # print(isCorrect)

print('Final Accuracy')
print(correct/tot)

print('Individual Accuracy')
print(net/net_count)

# print("Final")
# for i in range(100):
# print(segnumtot[i]/segnumcount[i])
