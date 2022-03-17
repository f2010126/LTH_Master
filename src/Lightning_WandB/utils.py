import os
from os import path
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule
from torch.nn.utils.prune import L1Unstructured, RandomUnstructured, global_unstructured, remove
import torch
from torch.nn.utils.prune import is_pruned
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar


def set_experiment_run(args):
    exp_dir = os.path.join(os.getcwd(), args.exp_dir)
    checkdir(exp_dir)
    trial_dir = path.join(exp_dir, args.wand_exp_name)
    checkdir(exp_dir)
    return trial_dir


def add_callbacks(args):
    call_list = [LearningRateMonitor(logging_interval="step"),
                 TQDMProgressBar(refresh_rate=100)]
    if args.early_stop:
        early_stopping = EarlyStopping('val_loss', patience=args.es_patience, mode='min', min_delta=args.es_delta,
                                       verbose=True)
        call_list.append(early_stopping)

    return call_list


def get_data_module(path, batch, seed=123, workers=0):
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

    # Data split 45k train, 5k val
    cifar10_dm = CIFAR10DataModule(
        data_dir=path,
        batch_size=batch,
        num_workers=workers,
        val_split=0.1,
        seed=seed,
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


def apply_prune(model, amt=0.0, prune_type='magnitude', global_prune=True):
    if global_prune:
        pruning_global(model, amt, prune_type)
    else:
        pruning_by_layer(model, amt, prune_type)


def pruning_by_layer(model, amt=0.0, prune_type='magnitude'):
    if prune_type == "magnitude":
        pruner = L1Unstructured(amt)
    elif prune_type == "random":
        pruner = RandomUnstructured(amt)
    # adds masks of all ones to each of the layers
    for n, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)) and n not in ['model.fc']:
            pruner.apply(m, name='weight', amount=amt)
        if isinstance(m, torch.nn.Linear) and n not in ['model.fc']:
            pruner.apply(m, name='weight', amount=0.0)


def pruning_global(model, amt=0.0, prune_type='magnitude'):
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)) and module_name not in [
            'model.fc']:
            parameters_to_prune.append((module, "weight"))

    if prune_type == 'random':
        pruner = RandomUnstructured
    elif prune_type == 'magnitude':
        pruner = L1Unstructured

    global_unstructured(
        parameters_to_prune,
        pruning_method=pruner,
        amount=amt,
    )


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
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)) and (
                f"{name}.weight_orig" in original_wgts.keys()):
            # do nothing for unpruned weights?
            if is_pruned(module) is False:
                continue
            with torch.no_grad():
                # copy the named params. no masks
                for param_name, param_val in module.named_parameters():
                    param_val.data.copy_(original_wgts[f"{name}.{param_name}"])


def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            # do nothing for unpruned weights?
            if is_pruned(module) is False:
                continue
            with torch.no_grad():
                remove(module,'weight')


'''
https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
'''


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
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
