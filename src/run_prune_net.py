from lenet import *
from data_and_augment import *
from run_lenet import run_training
from prune_model import prune_once
import torch.nn.utils.prune as prune

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def mask_weight(module, input):
#     module.weight = module.weight * module.mask


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
    # summary(model, (1, 28, 28),
    #         device='cuda' if torch.cuda.is_available() else 'cpu')
    prune_once(model, p_rate=0.0)
    # get hold of w0
    original_state_dict = model.state_dict()
    hooks = {}
    # register the pre hook
    # for name, module in model.named_modules():
    #     if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
    #         hooks[name] = module.register_forward_pre_hook(mask_weight)
    # Run and train the lenet OG, done in run_lenet.py
    metrics = run_training(model, n_epochs, batch)
    print(f"Original metrics {metrics}")
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pt")
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
            hooks[name] = module.register_forward_pre_hook(mask_weight)
    lastlayer = None
    for epoch in range(n_prune):
        # Prune and get the new mask.
        print(f"Number of Zeros: {countZeroWeights(model)}")
        masks = prune_once(model, p_rate=0.2)
        # create a dict that has the same keys as state dict w/o being linked to model.
        detached = dict([(name.replace(".weight_mask", ""), mask.clone()) for name, mask in masks])
        if lastlayer is not None:
            print(f"mask in lask layer same? {torch.equal(detached['fc3'], lastlayer)}")
        lastlayer = detached['fc3']
        # Load the OG weights and save the new mask to the layer
        # TODO: Freeze the 0 weights.
        model.load_state_dict(original_state_dict)
        for name, module in model.named_modules():
            if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
                mask_val = detached[name]
                module.mask = torch.nn.parameter.Parameter(mask_val, requires_grad=False)

        pruned_metrics = run_training(model, n_epochs, batch)
        print(f"Pruned metrics {metrics}")



if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    pruned(net, n_epochs=1, batch=128, n_prune=2)
    print("")
