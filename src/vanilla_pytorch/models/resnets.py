import torch
import torch.nn as nn
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Resnets(nn.Module):
    def __init__(self, in_channels, pretrained=False):
        super(Resnets, self).__init__()
        num_out_class = 10
        resnet18 = torchvision.models.resnet18(pretrained=pretrained, progress=True)
        resnet18.fc = nn.Linear(512, num_out_class)
        resnet18 = resnet18.to(device)
        self.model = resnet18

    # def apply(self,init_weights):
    #     self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
