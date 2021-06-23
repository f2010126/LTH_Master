from lenet import *
from convnets import *
from data_and_augment import *
from run_model import run_training
from prune_model import *
import argparse
from utils import *
import time

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
    all_masks = {key: mask.to(device) for key, mask in get_masks(model, p_rate=0)}
    original_state_dict = model.state_dict()
    # # incase loading happens
    # model_checkpt = torch.load("mnist_lenet_OG.pth")
    # model.load_state_dict(original_state_dict)
    # Run and train the lenet OG, done in run_model.py
    metrics, full_es = run_training(model, args=args)
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pth")

    return original_state_dict, all_masks, metrics['val_score'] * 100, full_es


def pruned(model, args):
    """
    Run the LTH experiment
    :param args: arguments from cmd or default values
    :param model: model to train
    :return: dictionary with pruning data
    """
    original_state_dict, all_masks, full_val, full_es = handle_OG_model(model, args)
    prune_data = [{"rem_weight": 100,
                   "val_score": full_val,
                   "full_es": full_es}]
    # init a random model
    in_chan = 1 if args.dataset == 'mnist' else 3
    rando_net = eval(args.model)(in_channels=in_chan)
    rando_net.apply(init_weights)
    for level in range(args.pruning_levels):
        # Prune and get the new mask. prune rate will vary with epoch.
        # TODO: IS THIS PRUNE RATE CORRECT??
        # prune_rate = args.pruning_rate ** (1 / (level+ 1))
        prune_rate = args.pruning_rate / 100
        masks = get_masks(model, p_rate=prune_rate)
        # create a dict that has the same keys as state dict w/o being linked to model.
        detached = dict([(name, mask.clone().to(device)) for name, mask in masks])
        update_masks(all_masks, detached)
        # Load the OG weights and mask it
        model.load_state_dict(original_state_dict)
        model = update_apply_masks(model, all_masks)
        # prune randomly inited model randomly
        prune_random(rando_net, prune_rate)
        non_zero = countRemWeights(model)
        print(f"Pruning round {level + 1} Weights remaining {non_zero} and 0% is {100 - non_zero}")
        last_run, pruned_es = run_training(model, args=args)
        rand_run, rand_es = run_training(rando_net, args)
        prune_data.append({"rem_weight": non_zero,
                           "val_score": last_run['val_score'] * 100,
                           "rand_init": rand_run['val_score'] * 100,
                           "pruned_es": pruned_es,
                           "rand_es": rand_es})
    # metrics
    return full_val, prune_data


if __name__ == '__main__':
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='LTH Experiments')
    parser.add_argument('-m', '--model', default='Net2',
                        help='Class name of model to train',
                        type=str, choices=['LeNet', 'Net2'])
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.0012,
                        help='learning rate 0.0012')

    parser.add_argument('--pruning-rate', type=int, default=20,
                        help='how much to prune. taken as a % (default: 20)')

    parser.add_argument('--pruning-levels', type=int, default=3,
                        help='No. of times to prune (default: 3), referred to as levels in paper')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    parser.add_argument('--early-stop', type=bool, default=False, help='Should Early stopping be done?')

    # prune to 30 to get 0.1% weights but 25 is ok too
    args = parser.parse_args()

    in_chan, img = (1, 28) if args.dataset == 'mnist' else (3, 32)
    # net = LeNet(in_channels=in_chan)
    net = eval(args.model)(in_channels=in_chan)
    net.apply(init_weights)
    summary(net, (in_chan, img, img),
            device='cuda' if torch.cuda.is_available() else 'cpu')
    baseline, pruned = pruned(net, args)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    json_dump = {"baseline": baseline, "prune_data": pruned}
    file_name = f"prune_{args.model}_{args.dataset}_{args.pruning_levels}"
    stored_at = save_data(json_dump, file_name + ".json")
    plot_graph(json_dump, file_at=file_name + ".png")
    print("")
