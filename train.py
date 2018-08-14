from data import *
from models import RagaDetector
from utils import Averager
from torch.utils.data import DataLoader
import multiprocessing
import progressbar
from tensorboardX import SummaryWriter
from bayes_opt import BayesianOptimization
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.set_default_tensor_type('torch.FloatTensor')

best_loss_model = None
best_accuracy_model = None

import pandas
df = pd.read_pickle('/home/sauhaarda/Dataset/song2raga.pkl')
song2ragaid = dict(zip(df.song_id, df.raga_id))

def transpose(chroma, ragaid, tonic, s):
    shift = random.randint(0, 11)
    if shift == 0:
        return chroma, ragaid, tonic, s

    return torch.cat([chroma[:, -shift:], chroma[:, :-shift]], 1), ragaid, (tonic + shift) % 12, s

def run(mode, dl, model, criterion, optimizer, val_songs):

    full_val_acc = {}
    if mode == 'train':
        model.train()
    else:
        for song in val_songs:
            full_val_acc[song] = torch.zeros(30)

        model.eval()

    print('Training:' if mode == 'train' else 'Validating:')

    with torch.set_grad_enabled(mode == 'train'):
        _loss = Averager()
        accuracy = Averager()
        tonic_accuracy = Averager()
        for batch_idx, (song, label, tonic, sid) in progressbar.progressbar(enumerate(dl), max_value=len(dl)):
            if mode == 'train' : optimizer.zero_grad()
            label = label.to(device)
            tonic = tonic.to(device)
            output = model(song.to(device))
            raga_loss = 0.9 * criterion(output[:, :30], label)
            tonic_loss = 0.1 * criterion(output[:, -12:], tonic)
            loss = raga_loss + tonic_loss
            if mode == 'train' : loss.backward()
            if mode == 'train' : optimizer.step()


            _, predicted = torch.max(output[:, :30].data, 1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            accuracy(float(correct)/float(total))

            if not mode == 'train':
                for batch in range(len(output[:, :30])):
                    full_val_acc[sid[batch]][predicted] += 0.1 

            _, predicted = torch.max(output[:, -12:].data, 1)
            total = label.size(0)
            correct = (predicted == tonic).sum().item()
            tonic_accuracy(correct/float(total))

            _loss(raga_loss.item())

        ans = 0
        if not mode == 'train':
            total_correct = 0
            for songres in full_val_acc:
                total_correct += int(torch.max(full_val_acc[songres].data, 0)[1].item() == song2ragaid[songres])
            ans = (float(total_correct)/len(full_val_acc))

        return _loss(), accuracy(), tonic_accuracy(), ans

def train_epochs(dropout, hidden_size, batch_size=120):
    writer = SummaryWriter('runs/full_net_acc_check4', comment='fixed tonic')

    tl, vl, val_songs = get_dataloaders(batch_size=batch_size, split=0.9166666, transform=transpose)
    model = torch.nn.DataParallel(RagaDetector(dropout, int(hidden_size)).to(device))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    writer.add_text(str(dropout), 'dropout')
    writer.add_text(str(hidden_size), 'hidden_size')
    writer.add_text(str(batch_size), 'batch_size')
    
    max_accuracy = 0
    for epoch in range(800):
        loss, accuracy, ta, _ = run('train', tl, model, criterion, optimizer, None)
        writer.add_scalar('data/train_loss', loss, epoch)
        writer.add_scalar('data/train_acc', accuracy, epoch)
        writer.add_scalar('data/train_t_acc', ta, epoch)

        loss, accuracy, ta, ans = run('val', vl, model, criterion, optimizer, val_songs)
        writer.add_scalar('data/val_loss', loss, epoch)
        writer.add_scalar('data/val_acc', accuracy, epoch)
        writer.add_scalar('data/val_t_acc', ta, epoch)
        writer.add_scalar('data/full_val_acc', ans, epoch)
        max_accuracy = max(accuracy, max_accuracy)

    return max_accuracy

if __name__ == '__main__':
    train_epochs(0.5, 256)
    # bo = BayesianOptimization(
    #     train_epochs,
    #     {'lr' : [0.00001, 0.1],
    #      'decay' : [0, 0.1],
    #      'dropout' : [0, 0.5],
    #      'hidden_size' : [128, 512]}
    # )
    # 
    # gp_params = {"alpha": 1e-5}
    # bo.maximize(n_iter=12, **gp_params)
