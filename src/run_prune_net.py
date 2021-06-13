from lenet import *
from data_and_augment import *
from run_lenet import run_training
from prune_model import get_masks, update_apply_masks
import torch.nn.utils.prune as prune

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def init_model_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
            masks[name] = torch.tensor(np.ones(module.weight.shape))

    return masks


# TODO: GET THIS LOGIC CHECKED
def update_masks(masks, new_mask):
    for name, mask in masks.items():
        print(f"non zeros in old mask: {torch.count_nonzero(masks[name])} and created mask {torch.count_nonzero(new_mask[name])}")
        masks[name] = mask * new_mask[name]
        print(f"non zeros in updated: {torch.count_nonzero(masks[name])}")


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
    # get hold of w0
    original_state_dict = model.state_dict()

    # Run and train the lenet OG, done in run_lenet.py
    metrics = run_training(model, n_epochs, batch)
    print(f"Original metrics {metrics}")
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pt")

    all_masks = init_model_masks(model)
    for epoch in range(n_prune):
        # Prune and get the new mask. prune rate will vary with epoch.
        # TODO: IS THIS PRUNE RATE CORRECT??
        masks = get_masks(model, p_rate=(20 ** (1 / (epoch + 1))) / 100)
        # create a dict that has the same keys as state dict w/o being linked to model.
        detached = dict([(name.replace(".weight_mask", ""), mask.clone()) for name, mask in masks])
        update_masks(all_masks, detached)
        # Load the OG weights and mask it
        for name, module in model.named_modules():
            if any([isinstance(module, cl) for cl in [nn.Conv2d, nn.Linear]]):
                prune.remove(module, name='weight')
        model.load_state_dict(original_state_dict)
        # apply combined masks here
        model = update_apply_masks(model, all_masks)
        print(f"Zero weights {countZeroWeights(model)}")
        pruned_metrics = run_training(model, n_epochs, batch)
        print(f"Pruned metrics {pruned_metrics}")


if __name__ == '__main__':
    net = LeNet()
    net.apply(init_weights)
    pruned(net, n_epochs=40, batch=128, n_prune=3)
    print("")
