from lenet import *
from data_and_augment import *
from run_lenet import run_training
from prune_model import prune_once
import torch.nn.utils.prune as prune

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def mask_weight(module, input, output):
    module.weight = module.weight * module.mask


def pruned(model, n_epochs=5000, batch=128, n_prune=3):
    """
    Run the LTH experiment
    :param model:
    :param n_epochs: numebr of epochs to train for
    :param batch: batch size
    :param n_prune: number of times we prune-mask-retrain
    :return:
    """
    model = model.to(device)
    prune_once(model, p_rate=0.0)
    # get hold of w0
    original_state_dict = model.state_dict()
    hooks = {}
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
            hooks[name] = module.register_forward_hook(mask_weight)
    # Run and train the lenet OG, done in run_lenet.py
    run_training(model, n_epochs, batch)
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pt")

    for epoch in range(n_prune):
        # Prune and get the new mask.
        print(f"Number of Zeros: {countZeroWeights(model)}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TRAINABLE :{trainable}")
        masks = prune_once(model)
        # create a dict that has the same keys as state dict w/o being linked to model.
        detached = dict([(name.replace(".weight_mask", ""), mask.clone()) for name, mask in masks])
        # Load the OG weights and save the new mask to the layer
        # TODO: Freeze the 0 weights.
        model.load_state_dict(original_state_dict)
        for name, module in model.named_modules():
            if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
                mask_val = detached[name]
                module.mask = torch.nn.parameter.Parameter(mask_val, requires_grad=False)

        run_training(model, n_epochs, batch)


if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    pruned(net, n_epochs=1, batch=128, n_prune=3)
    print("")
