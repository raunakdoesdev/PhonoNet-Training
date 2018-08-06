import torch.nn as nn
import torch
from collections import OrderedDict


class RagaDetector(nn.Module):
    def __init__(self, dropout=0.15, hidden_size=256):
        super(RagaDetector, self).__init__()

        self.norm0 = nn.BatchNorm2d(1)
        self.encoder = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(1)),

            ('conv1', nn.Conv2d(1, 64, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(64)),
            ('elu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool2d([1, 2])),
            ('drop1', nn.Dropout(p=dropout)),

            ('conv2', nn.Conv2d(64, 128, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(128)),
            ('elu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d([1, 3])),
            ('drop2', nn.Dropout(p=dropout)),

            ('conv3', nn.Conv2d(128, 128, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(128)),
            ('elu3', nn.LeakyReLU()),
            ('pool3', nn.MaxPool2d([2, 4])),
            ('drop3', nn.Dropout(p=dropout)),

            ('conv4', nn.Conv2d(128, 128, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(128)),
            ('elu4', nn.LeakyReLU()),
            ('pool4', nn.MaxPool2d([4, 4])),
            ('drop4', nn.Dropout(p=dropout))
        ]))

        self.fc1 = nn.Linear(3968, hidden_size)
        # self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 30)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x.transpose(0, 1).unsqueeze(1)
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.fc2(x)
        return x
