from lenet import *
from data_and_augment import *
from run_lenet import run_training
from prune_model import get_masks, update_apply_masks
import torch.nn.utils.prune as prune
import argparse
from utils import *

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
        masks[name] = torch.logical_and(mask,new_mask[name])
        # masks[name] = mask * new_mask[name]
        print(f"non zeros in updated: {torch.count_nonzero(masks[name])}")


def pruned(model, args):
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
    metrics, _ = run_training(model,args=args)
    print(f"Original metrics {metrics}")
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pt")

    all_masks = init_model_masks(model)
    prune_data=[]
    for epoch in range(args.pruning_epochs):
        # Prune and get the new mask. prune rate will vary with epoch.
        # TODO: IS THIS PRUNE RATE CORRECT??
        masks = get_masks(model, p_rate=(args.pruning_rate ** (1 / (epoch + 1))) / 100)
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
        per_zero, _ = countZeroWeights(model)
        print(f"Zero weights {per_zero}")
        last_run, pruned_metrics = run_training(model, args=args)
        print(f"Pruned metrics {pruned_metrics}")
        prune_data.append({"rem_weight":100-per_zero,
                           "val_score":last_run['val_score']})
    print(prune_data)
    return metrics['val_score'],prune_data


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='LTH MNSIT LeNet')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate 0.005')
    parser.add_argument('--pruning-rate', type=int, default=20,
                        help='how much to prune. taken as a % (default: 20)')
    parser.add_argument('--pruning-epochs', type=int, default=10,
                        help='No. of times to prune (default: 3)')
    args = parser.parse_args()

    net = LeNet()
    net.apply(init_weights)
    baseline, pruned = pruned(net,args)
    plot_graph(baseline, pruned)
    print("")
