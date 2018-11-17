from data import *
from models import RagaDetector
from utils import Averager
from torch.utils.data import DataLoader
import multiprocessing
import progressbar
from tensorboardX import SummaryWriter
from bayes_opt import BayesianOptimization
import random
import gc


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
            song = song.to(device)
            output = model(song)
            raga_loss =  criterion(output[:, :30], label)
            # tonic_loss = 0.1 * criterion(output[:, -12:], tonic)
            loss = raga_loss
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

def train_epochs(dropout, hidden_size, batch_size=1):
    run_name = 'aug1'
    writer = SummaryWriter('runs/' + run_name, comment='fixed hidden size 1')

    bacc = 0.0  # initialize best scores
    bloss = 1000000  # initialize best scores

    # tl, vl, val_songs = get_dataloaders(batch_size=batch_size, split=.95, transform=transpose)
    tl, vl, val_songs = get_dataloaders(0, batch_size=batch_size, transform=transpose)

    model = RagaDetector(dropout, int(hidden_size)).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    writer.add_text(str(dropout), 'dropout')
    writer.add_text(str(hidden_size), 'hidden_size')
    writer.add_text(str(batch_size), 'batch_size')
    
    max_accuracy = 0
    for epoch in range(800):
        print(epoch)
        loss, accuracy, ta, _ = run('train', tl, model, criterion, optimizer, None)
        writer.add_scalar('data/train_loss', loss, epoch)
        writer.add_scalar('data/train_acc', accuracy, epoch)
        writer.add_scalar('data/train_t_acc', ta, epoch)

        loss, accuracy, ta, ans = run('val', vl, model, criterion, optimizer, val_songs)
        writer.add_scalar('data/val_loss', loss, epoch)
        writer.add_scalar('data/val_acc', accuracy, epoch)
        writer.add_scalar('data/val_t_acc', ta, epoch)
        writer.add_scalar('data/full_val_acc', ans, epoch)

        print(loss)
        if accuracy > bacc:
            bacc = accuracy
            torch.save({'net' : model.state_dict(), 'epoch' : epoch, 'loss' : loss, 'acc' : accuracy, 'val_songs' : val_songs}, 'saves/0/{}_epoch_{}_acc_{}.model'.format(run_name, epoch, accuracy))
        if loss < bloss:
            print('saving a loss model!')
            print('bloss')
            bloss = loss
            torch.save({'net' : model.state_dict(), 'loss' : loss, 'epoch' : epoch, 'acc' : accuracy, 'val_songs' : val_songs}, 'saves/0/{}_epoch_{}_loss_{}.model'.format(run_name, epoch, loss))
        bloss = min(bloss, loss)
            
        max_accuracy = max(accuracy, max_accuracy)

    return max_accuracy

if __name__ == '__main__':
    train_epochs(0.6, 256, 120)
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
