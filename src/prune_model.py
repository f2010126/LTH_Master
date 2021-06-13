import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from lenet import *


def get_masks(model, p_rate=0.2, prune_amts={}):
    """
    prune the lowest p% weights by magnitude per layer

    :param model: model to prune
    :param p_rate: prune rate = 0.2 as per paper
    :param prune_amts: dictionary
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


def update_apply_masks(model, masks):
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
            module = prune.custom_from_mask(module, name='weight', mask=masks[name])
            prune.remove(module, "weight")
            print("")
    return model


def prune_tut(net):
    module = net.conv1  # has the mask from my addition
    prune.custom_from_mask(module, name='bias', mask=torch.Tensor([0., 1., 1., 0., 0., 1.]))
    prune.custom_from_mask(module, name='bias', mask=torch.Tensor([0., 1., 0., 1., 0., 1.]))
    print(module._forward_pre_hooks)
    prune.remove(module, "bias")
    prune.l1_unstructured(module, name="bias", amount=2)
    prune.l1_unstructured(module, name="bias", amount=1)
    prune.l1_unstructured(module, name="bias", amount=1)
    prune.remove(module, "bias")


if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    prune_tut(net)
    masks = get_masks(net)
    print(f"Count zero : {countZeroWeights(net)}")
