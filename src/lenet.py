import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torch


def num_flat_features(x):
    """
    Get the number of features in a batch of tensors `x`.
    Needed for torch summary
    """
    size = x.size()[1:]
    return np.prod(size)


class LeNet(nn.Module):
    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        One forward pass through the network.

        Args:
            x: input
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # gaussian glorot init
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


if __name__ == '__main__':
    net = LeNet()
    print(summary(net, (1, 28, 28)))
