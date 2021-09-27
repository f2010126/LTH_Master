import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from src.vanilla_pytorch.utils import init_weights


class Net2(nn.Module):
    def __init__(self, in_channels=1):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    net = Net2(in_channels=3)
    net.apply(init_weights)
    summary(net, (3, 32, 32),
            device='cuda' if torch.cuda.is_available() else 'cpu')
