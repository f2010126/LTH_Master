import os
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules import CIFAR10DataModule
from torch.nn.utils.prune import L1Unstructured
import torch
from torch.nn.utils.prune import is_pruned


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
            pruner.apply(m, name='weight', amount=amt)


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
                    param_val.param.data.copy_(original_wgts[f"{name}.{param_name}"])


def check_model_change(prev_iter_dict, model):
    for name, param in model.named_parameters():
        prev_param = prev_iter_dict[name]
        assert not torch.allclose(prev_param, param), 'model not updating'
