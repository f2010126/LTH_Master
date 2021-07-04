import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torch
from utils import init_weights


def num_flat_features(x):
    """
    Get the number of features in a batch of tensors `x`.
    Needed for torch summary
    """
    size = x.size()[1:]
    return np.prod(size)

    # gaussian glorot init


def print_weights(model):
    """
    Print the weights of the model
    :param model:
    :return:
    """
    for param in model.parameters():
        print(param.data)


class LinearNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        if in_channels == 1:
            features = 784
        elif in_channels == 3:
            features = 3072
        self.fc1 = nn.Linear(in_features=features, out_features=10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc1(x.view(x.shape[0], -1)))


class LeNet(nn.Module):
    # network structure
    def __init__(self, in_channels=1):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(in_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723

    def forward(self, x):
        """
        One forward pass through the network.

        Args:
            x: input
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, num_flat_features(x))  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet300(nn.Module):
    """
    2 FC layers with 300, 100 units
    """

    def __init__(self, in_channels=1):
        super(LeNet300, self).__init__()
        if in_channels == 1:
            features = 784
        elif in_channels == 3:
            features = 3072
        self.fc1 = nn.Linear(in_features=features, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.output = nn.Linear(in_features=100, out_features=10)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return self.output(out)



if __name__ == '__main__':
    in_chan = 1
    net = LeNet300(in_channels=in_chan)
    net.apply(init_weights)
    summary(net, (in_chan, 32, 32),
            device='cuda' if torch.cuda.is_available() else 'cpu')

