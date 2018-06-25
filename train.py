from data import *
from models import RagaDetector
from torch.utils.data import DataLoader

bptt = 500


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def time_split(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    return data


torch.set_default_tensor_type('torch.FloatTensor')

if __name__ == '__main__':
    ds = RagaDataset('/home/sauhaarda/Dataset')
    train_loader = DataLoader(
        ds,
        batch_size=3,
        num_workers=2,
        shuffle=True,
        collate_fn=PadCollate())
    model = RagaDetector()

    for epoch in range(100):
        for song, label in train_loader:
            hidden = model.init_hidden(song.size(1))
            for batch, i in enumerate(range(0, song.size(0) - 1, bptt)):
                hidden = repackage_hidden(hidden)
                x = time_split(song, i)
                x, hidden = model(x, hidden)
                print('End size: {}'.format(x.shape))
