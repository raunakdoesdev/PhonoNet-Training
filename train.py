from data import *
from models import RagaDetector
from utils import Averager
from torch.utils.data import DataLoader
import multiprocessing
import progressbar
from tensorboardX import SummaryWriter
from bayes_opt import BayesianOptimization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.set_default_tensor_type('torch.FloatTensor')

def run(mode, dl, model, criterion, optimizer):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    print('Training:' if mode == 'train' else 'Validating:')

    with torch.set_grad_enabled(mode == 'train'):
        _loss = Averager()
        accuracy = Averager()
        for batch_idx, (song, label) in progressbar.progressbar(enumerate(dl), max_value=len(dl)):
            if mode == 'train' : optimizer.zero_grad()
            label = label.to(device)
            output = model(song.to(device))
            loss = criterion(output, label)
            if mode == 'train' : loss.backward()
            if mode == 'train' : optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            accuracy(float(correct)/float(total))

            loss = criterion(output, label.to(device))
            _loss(loss.item())

        return _loss(), accuracy()

def train_epochs(dropout, hidden_size, batch_size=120):
    writer = SummaryWriter('runs/split_gpa_2', comment='')

    tl, vl = get_dataloaders(batch_size=batch_size, split=0.7)
    model = torch.nn.DataParallel(RagaDetector(dropout, int(hidden_size)).to(device))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    writer.add_text(str(dropout), 'dropout')
    writer.add_text(str(hidden_size), 'hidden_size')
    writer.add_text(str(batch_size), 'batch_size')
    
    max_accuracy = 0
    for epoch in range(800):
        loss, accuracy = run('train', tl, model, criterion, optimizer)
        writer.add_scalar('data/train_loss', loss, epoch)
        writer.add_scalar('data/train_acc', accuracy, epoch)

        loss, accuracy = run('val', vl, model, criterion, optimizer)
        writer.add_scalar('data/val_loss', loss, epoch)
        writer.add_scalar('data/val_acc', accuracy, epoch)
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
