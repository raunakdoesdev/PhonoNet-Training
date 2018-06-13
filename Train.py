from RagaDataset import *
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)  # 16 filters
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1) # 32 filters
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        print(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        print(x)
        print(x.size())

dataset = RagaDataset('/home/sauhaarda/Dataset', 1)
network = Network()
network.forward(dataset[0][0].transpose(0, 1)[0:100].unsqueeze(0).unsqueeze(0).float())
# network.forward(dataset[0][0].unsqueeze(0).unsqueeze(0).float())
