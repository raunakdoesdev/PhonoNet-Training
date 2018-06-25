import torch.nn as nn
import torch
from collections import OrderedDict


class RagaDetector(nn.Module):
    def __init__(self):
        super(RagaDetector, self).__init__()

        self.norm0 = nn.BatchNorm2d(1)
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

        self.lstm = nn.LSTM(128, 72, num_layers=2, dropout=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden):
        x = x.transpose(0, 1).unsqueeze(1)
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = x.transpose(0, 2).transpose(1, 2).view(-1, batch_size, 128)
        x, hidden = self.lstm(x, hidden)

        return x, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(2, bsz, 72),
                weight.new_zeros(2, bsz, 72))
