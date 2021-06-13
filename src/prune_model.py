import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from lenet import *


def prune_once(model, p_rate=0.2):
    """
    prune the lowest p% weights by magnitude per layer

    :param model: model to prune
    :param p_rate: prune rate = 0.2 as per paper
    :return: the created mask. model has served it's purpose.
    """
    # TODO: Adjust pruning
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            module = prune.l1_unstructured(module, name='weight', amount=p_rate)
        # prune 90% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            module = prune.l1_unstructured(module, name='weight', amount=p_rate)
    return list(model.named_buffers())


def sameModel(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    prune_once(net, p_rate=0.0)
    original_state_dict = net.state_dict()
    torch.save(net.state_dict(), "prune_lenet_OG.pt")

    masks = prune_once(net, p_rate=1.0)
    # detached = [(name, masks[0][1].clone().detach()) for name, mask in masks]
    detached = [(name, masks[0][1].clone()) for name, mask in masks]
    for (name, mask) in masks:
        print(name)
    net.load_state_dict(original_state_dict)
