from lenet import *
from data_and_augment import *
from run_lenet import run_training
from prune_model import get_masks, update_apply_masks
import argparse
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: GET THIS LOGIC CHECKED
def update_masks(masks, new_mask):
    """
    Combine the new mask
    :param masks: masks so far
    :param new_mask: new mask from pruned
    """
    for name, mask in masks.items():
        masks[name] = torch.logical_and(mask, new_mask[name])


def handle_OG_model(model, args):
    """
    Run, train, setup and save the baseline for the experiment
    :param model: baseline model, weights inited with glorot gaussian
    :param args: HP to use while training
    :return: the original weights of the network, initial masks
    """
    # get hold of w0
    all_masks = dict(get_masks(model, p_rate=0))
    original_state_dict = model.state_dict()
    # # incase loading happens
    # model_checkpt = torch.load("mnist_lenet_OG.pth")
    # model.load_state_dict(original_state_dict)
    # Run and train the lenet OG, done in run_lenet.py
    metrics, _ = run_training(model, args=args)
    print(f"Original metrics {metrics}")
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pth")
    non_zero = countRemWeights(model)
    print(f"Weights remaining in baseline {non_zero}%")

    return original_state_dict, all_masks


def pruned(model, args):
    """
    Run the LTH experiment
    :param args: arguments from cmd or default values
    :param model: model to train
    :return: dictionary with pruning data
    """
    original_state_dict, all_masks = handle_OG_model(model, args)

    prune_data = []
    for level in range(args.pruning_levels):
        # Prune and get the new mask. prune rate will vary with epoch.
        # TODO: IS THIS PRUNE RATE CORRECT??
        # prune_rate = args.pruning_rate ** (1 / (epoch + 1))
        masks = get_masks(model, p_rate=args.pruning_rate / 100)
        # create a dict that has the same keys as state dict w/o being linked to model.
        detached = dict([(name, mask.clone().to(device)) for name, mask in masks])
        update_masks(all_masks, detached)
        # Load the OG weights and mask it
        model.load_state_dict(original_state_dict)
        # apply combined masks here
        model = update_apply_masks(model, all_masks)
        non_zero = countRemWeights(model)
        print(f"Weights remaining {non_zero}%")
        last_run, pruned_metrics = run_training(model, args=args)
        prune_data.append({"rem_weight": non_zero,
                           "val_score": last_run['val_score']})
    print(prune_data)
    # metrics
    return last_run['val_score'], prune_data


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='LTH MNSIT LeNet')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate 0.005')

    parser.add_argument('--pruning-rate', type=int, default=20,
                        help='how much to prune. taken as a % (default: 20)')

    parser.add_argument('--pruning-levels', type=int, default=8,
                        help='No. of times to prune (default: 3), referred to as levels in paper')
    args = parser.parse_args()

    net = LeNet()
    net.apply(init_weights)
    baseline, pruned = pruned(net, args)
    plot_graph(baseline, pruned)
    print("")
