import torch.nn.utils.prune as prune
import torch
from utils import count_rem_weights
from linearnets import LeNet, init_weights


def get_masks(model, p_rate=0.2, prune_amts=None):
    """
    prune the lowest p% weights by magnitude per layer

    :param model: model to prune
    :param p_rate: prune rate = 0.2 as per paper
    :param prune_amts: dictionary
    :return: the created mask. model has served it's purpose.
    """
    # TODO: Adjust pruning with output layer
    if prune_amts is None:  # ie dict is empty, use the default prune rate
        prune_amts = {"linear": p_rate, "conv": p_rate, "last": p_rate}

    for i, (name, module) in enumerate(model.named_children()):
        # prune 10% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            module = prune.l1_unstructured(module, name='weight', amount=prune_amts['conv'])
        # prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            module = prune.l1_unstructured(module, name='weight', amount=prune_amts['linear'])

    return list(model.named_buffers())


def update_apply_masks(model, masks):  # doesn't seem to be needed.
    for key, val in masks.items():
        layer = getattr(model, key.split('.')[0])
        layer.weight_mask = val
    # for name, module in model.named_children():
    #     if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
    #         module = prune.custom_from_mask(module, name='weight', mask=masks[name + ".weight_mask"])
    return model


def prune_random(model, p_rate=0.2, prune_amts=None):
    if prune_amts is None:  # ie dict is empty, use the default prune rate
        prune_amts = {"linear": p_rate, "conv": p_rate, "last": p_rate}
    for name, module in model.named_children():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            module = prune.random_unstructured(module, name='weight', amount=prune_amts['conv'])
        # prune 90% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            module = prune.random_unstructured(module, name='weight', amount=prune_amts['linear'])


def prune_tut(net):
    module = net.conv1  # has the mask from my addition
    prune.custom_from_mask(module, name='bias', mask=torch.Tensor([0., 1., 1., 0., 0., 1.]))
    prune.custom_from_mask(module, name='bias', mask=torch.Tensor([0., 1., 0., 1., 0., 1.]))
    print(module._forward_pre_hooks)
    prune.remove(module, "bias")
    prune.l1_unstructured(module, name="bias", amount=2)


if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    prune_rate = 0.8
    prune_custom = {"linear": 0.2, "conv": 0.1, "last": 0.1}
    masks = get_masks(net, p_rate=prune_rate)
    print(f"Count zero : {count_rem_weights(net)}")
