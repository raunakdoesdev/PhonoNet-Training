from data import *
from models import RagaDetector
from utils import Averager
from torch.utils.data import DataLoader
import multiprocessing
import progressbar
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

torch.set_default_tensor_type('torch.FloatTensor')
if __name__ == '__main__':
    ds = RagaDataset('/home/sauhaarda/Dataset')
    train_loader = DataLoader(
        ds,
        batch_size=10,
        num_workers=multiprocessing.cpu_count(),
        shuffle=True)
    model = RagaDetector().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001/4)
    
    n_iter = 0
    for epoch in range(100):
        epoch_loss = Averager()
        for batch_idx, (song, label) in enumerate(train_loader):
            batch_loss = Averager()
            optimizer.zero_grad()
            output = model(song.to(device))
            loss = criterion(output, label.to(device))
            loss.backward()
            batch_loss(loss.item())
            optimizer.step()
            epoch_loss(batch_loss())
            writer.add_scalar('data/batch_loss', batch_loss(), batch_idx)
        writer.add_scalar('data/epoch_loss', epoch_loss(), epoch)
