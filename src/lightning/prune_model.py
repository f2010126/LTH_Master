import torch.nn.utils.prune as prune
import torch

def count_rem_weights(model):
    """
    Percetage of weights that remain for training
    :param model:
    :return: % of weights remaining
    """
    total_weights = 0
    rem_weights = 0
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [torch.nn.Conv2d, torch.nn.Linear]]):
            rem_weights += torch.count_nonzero(module.weight)
            total_weights += sum([param.numel() for param in module.parameters()])
    # return % of non 0 weights
    return rem_weights.item() / total_weights * 100

def update_masks(masks, new_mask):
    """
    Combine the new mask
    :param masks: masks so far
    :param new_mask: new mask from pruned
    """
    for name, mask in masks.items():
        masks[name] = torch.logical_and(mask, new_mask[name])

def get_masks(model, prune_amts=None):
    """
    prune the lowest p% weights by magnitude per layer

    :param model: model to prune
    :param p_rate: prune rate = 0.2 as per paper
    :param prune_amts: dictionary
    :return: the created mask. model has served it's purpose.
    """
    if prune_amts is None:  # ie dict is empty, use the default prune rate = 0.2
        prune_amts = {"linear": 0.2, "conv": 0.2, "last": 0.2}
    print(f"before pruning {count_rem_weights(model)}")
    for i, (name, module) in enumerate(model.named_modules()):
        # name and val
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            module = prune.l1_unstructured(module, name='weight', amount=prune_amts['conv'])
        # prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            module = prune.l1_unstructured(module, name='weight', amount=prune_amts['linear'])
    print(f"pruning should ve done {count_rem_weights(model)}")
    return list(model.named_buffers())


def update_apply_masks(model, masks):  # doesn't seem to be needed.
    for key, val in masks.items():
        layer = getattr(model, key.split('.')[0])
        layer.weight_mask = val
    # for name, module in model.named_children():
    #     if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
    #         module = prune.custom_from_mask(module, name='weight', mask=masks[name + ".weight_mask"])
    return model


def prune_random(model, prune_amts=None):
    if prune_amts is None:  # ie dict is empty, use the default prune rate =0.2
        prune_amts = {"linear": 0.2, "conv": 0.2, "last": 0.2}
    for name, module in model.named_children():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            module = prune.random_unstructured(module, name='weight', amount=prune_amts['conv'])
        # prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            module = prune.random_unstructured(module, name='weight', amount=prune_amts['linear'])

