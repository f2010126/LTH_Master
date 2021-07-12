import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import init_weights


class Net2(nn.Module):
    def __init__(self, in_channels=1):
        super(Net2, self).__init__()
        # TODO: sort into conv and fc blocks
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # dropout after pooling
        x = torch.flatten(x, 1)
        # FC_ block
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # dropout after a layer.
        output = self.output_layer(x)
        return output


class Net4(nn.Module):
    def __init__(self, in_channels=1):
        super(Net4, self).__init__()
        # TODO: sort into conv and fc blocks
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1))
        self.conv4 = nn.Conv2d(128, 128, (3, 3), (1, 1))
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # dropout after pooling
        x = torch.flatten(x, 1)
        # FC_ block
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # dropout after a layer.
        output = self.output_layer(x)
        return output


class Net4Drop(nn.Module):
    def __init__(self, in_channels=1):
        super(Net4Drop, self).__init__()
        # TODO: sort into conv and fc blocks
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1))
        self.conv4 = nn.Conv2d(128, 128, (3, 3), (1, 1))
        self.dropout_fc = nn.Dropout(0.5)  # for FC
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # dropout after pooling
        x = torch.flatten(x, 1)
        # FC_ block
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        # dropout after a layer.
        output = self.output_layer(x)
        return output


if __name__ == '__main__':
    net = Net4Drop(in_channels=3)
    net.apply(init_weights)
    summary(net, (3, 32, 32),
            device='cuda' if torch.cuda.is_available() else 'cpu')
