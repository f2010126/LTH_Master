import os
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule
from torch.nn.utils.prune import L1Unstructured
import torch
from torch.nn.utils.prune import is_pruned
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def get_data_module(path, batch, workers=0):
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ])
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=path,
        batch_size=batch,
        num_workers=workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    return cifar10_dm


# Check if the directory exist and if not, create a new directory
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# List Layers in Model
def layer_looper(model):
    for name, param in model.named_parameters():
        print(name, param.size())


def check_pruned_linear(module):
    """Check if module was pruned.
    ----------
    module : module containing a bias and weight

    Returns
    -------
    bool
        True if the model has been pruned.
    """
    params = {param_name for param_name, _ in module.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}
    return params == expected_params


def apply_pruning(model, amt=0.0):
    pruner = L1Unstructured(amt)
    # adds masks of all ones to each of the layers
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            pruner.apply(m, name='weight', amount=amt)
        if isinstance(m, torch.nn.Linear):
            pruner.apply(m, name='weight', amount=0.0)


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


# PRUNING FUNCTIONS #
# For the forward pass to work without modification,
# the weight attribute needs to exist. weight= mask*weight_orig in forward hook
# So weight_orig needs to be changed.
def reset_weights(model, original_wgts):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and (
                f"{name}.weight_orig" in original_wgts.keys()):
            # do nothing for unpruned weights?
            if is_pruned(module) is False:
                continue
            with torch.no_grad():
                # copy the named params. no masks
                for param_name, param_val in module.named_parameters():
                    param_val.data.copy_(original_wgts[f"{name}.{param_name}"])


def check_model_change(prev_iter_dict, model):
    for name, param in model.named_parameters():
        prev_param = prev_iter_dict[name]
        assert not torch.allclose(prev_param, param), 'model not updating'

'''
https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
'''
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    fig = plt.get_figure()
    return fig
