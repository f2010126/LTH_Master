import argparse
import time
import LTH_Constants
import torch
import copy
from torchsummary import summary
from linearnets import LeNet, LinearNet
from convnets import Net2
from run_model import run_training
from prune_model import get_masks, update_apply_masks
from prune_model import prune_random
from utils import save_data, plot_graph
from utils import init_weights, countRemWeights

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
    all_masks = {key: mask.to(device) for key, mask in get_masks(model, prune_amts=LTH_Constants.init_mask)}
    original_state_dict = copy.deepcopy(model.state_dict())
    # # incase loading happens
    # model_checkpt = torch.load("mnist_lenet_OG.pth")
    # model.load_state_dict(original_state_dict)
    # Run and train the lenet OG, done in run_model.py
    metrics, full_es, _ = run_training(model, device, args=args)
    # Save OG model
    torch.save(model.state_dict(), "mnist_lenet_OG.pth")

    return original_state_dict, all_masks, {"val_score": metrics['val_score'] * 100,
                                            "full_es": full_es}


def pruned(model, args):
    """
    Run the LTH experiment
    :param args: arguments from cmd or default values
    :param model: model to train
    :return: dictionary with pruning data
    """
    original_state_dict, all_masks, baselines = handle_OG_model(model, args)
    prune_data = []
    # init a random model
    in_chan = 1 if args.dataset == 'mnist' else 3
    rando_net = globals()[args.model](in_channels=in_chan)
    rando_net.apply(init_weights)
    # set pruning configs
    prune_amt = LTH_Constants.conv2_prune if args.model == 'Net2' else LTH_Constants.lenet_prune
    for level in range(args.pruning_levels):
        # Prune and get the new mask.
        with torch.no_grad():
            prune_rate = args.pruning_rate / 100
            masks = get_masks(model, prune_amts=prune_amt)
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
        last_run, pruned_es, training = run_training(model, device, args=args)
        # rand_run, rand_es = {'val_score':0}, 0
        rand_run, rand_es, _ = run_training(rando_net, device, args)
        prune_data.append({"rem_weight": non_zero,
                           "val_score": last_run['val_score'] * 100,
                           "rand_init": rand_run['val_score'] * 100,
                           "pruned_es": pruned_es,
                           "rand_es": rand_es,
                           "training_data": training})
    # metrics
    # TODO: baseline
    return baselines, prune_data


if __name__ == '__main__':
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='LTH Experiments')
    parser.add_argument('--model', default='Net2',
                        help='Class name of model to train',
                        type=str, choices=['LeNet', 'Net2', 'LinearNet'])
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
    parser.add_argument('--early-stop',
                        action='store_true', help='Does Early if enabled')
    parser.add_argument('--early-delta', type=float, default=0.0005,
                        help='Difference b/w best and current to decide to stop early')
    # prune to 30 to get 0.1% weights but 25 is ok too
    args = parser.parse_args()

    in_chan, img = (1, 32) if args.dataset == 'mnist' else (3, 32)
    net = globals()[args.model](in_channels=in_chan).to(device)
    net.apply(init_weights)
    summary(net, (in_chan, img, img),
            device=device.type)
    run_data, pruned_data = pruned(net, args)
    run_data["prune_data"] = pruned_data
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    file_name = f"prune_{args.model}_{args.dataset}_{args.pruning_levels}"
    stored_at = save_data(run_data, file_name + ".json")
    plot = LTH_Constants.default_plot_es
    plot['title'] = file_name
    plot['y_max'] = args.epochs
    plot_graph(run_data, plot, file_at=file_name + "_es.png")
    plot = LTH_Constants.default_plot_acc
    plot['title'] = file_name
    plot['y_max'] = 100
    plot_graph(run_data, plot, file_at=file_name + ".png")
    print(f"Files saved at {stored_at}")
