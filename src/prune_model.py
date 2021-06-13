import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from lenet import *

def check_hook(module):
    for hook in module._forward_pre_hooks.values():
        print(hook)
        # if hook.__name__ == "weight":  # select out the correct hook
        #     break

def prune_once(model, p_rate=0.5, prune_amts={}):
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

        print(f"module {name} has {len(module._forward_pre_hooks)}")
    return list(model.named_buffers())


def sameModel(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def prune_tut(net):
    module = net.conv1
    prune.random_unstructured(module, name="weight", amount=0.3)
    print(list(module.named_buffers())) # has the mask from my addition
    print(module._forward_pre_hooks)
    prune.l1_unstructured(module, name="bias", amount=2)
    prune.l1_unstructured(module, name="bias", amount=1)
    prune.l1_unstructured(module, name="bias", amount=1)
    print("")


if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    prune_tut(net)
    prune_once(net, p_rate=0.0)
    original_state_dict = net.state_dict()
    print(f"Count zero : {countZeroWeights(net)}")
    masks = prune_once(net)
    print(f"Count zero : {countZeroWeights(net)}")
    detached = [(name, masks[0][1].clone()) for name, mask in masks]

