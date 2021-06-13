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

    # gaussian glorot init


def init_weights(m):
    """
        Initialise weights acc the Xavier initialisation and bias set to 0.01
        :param m:
        :return:
        """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def print_weights(model):
    """
    Print the weights of the model
    :param model:
    :return:
    """
    for param in model.parameters():
        print(param.data)


def countZeroWeights(model):
    zeros = 0
    total_weights = 0
    for name, param in model.named_parameters():
        if param is not None and "bias" not in name:
            zeros += torch.sum((param == 0).int()).data.item()
        total_weights += param.numel()
    # return % of 0 weights
    return (zeros / total_weights * 100)


class LeNet(nn.Module):
    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.create_mask()

    # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
    def create_mask(self):
        """
        Create the initial mask here and register it as a buffer
        :return:
        """
        # TODO: are biases masked?
        print(f"REINIT??")
        for name, module in self.named_modules():
            if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
                mask = torch.tensor(np.ones(module.weight.shape))
                module.register_buffer("mask", torch.nn.parameter.Parameter(mask, requires_grad=False))
            # if isinstance(module, torch.nn.Conv2d):
            #     mask = torch.tensor(np.ones(module.weight.shape))
            #     module.register_parameter("mask", torch.nn.parameter.Parameter(mask, requires_grad=False))
            # elif isinstance(module, torch.nn.Linear):
            #     mask = torch.tensor(np.ones(module.weight.shape))
            #     module.register_parameter("mask", torch.nn.parameter.Parameter(mask, requires_grad=False))

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

    def freeze_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print("")

if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    net.fc2.weight = torch.nn.Parameter(torch.zeros(net.fc2.weight.shape))
    print(f"The number of zeros in here after I zero'd out fc2: {countZeroWeights(net)}")
    # net.freeze_weights()
    #print(summary(net, (1, 28, 28)))
