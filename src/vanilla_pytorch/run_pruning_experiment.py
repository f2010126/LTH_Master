import argparse
import time
import torch
import copy
# from torchsummary import summary
from src.vanilla_pytorch.LTH_Constants import default_plot_es, default_plot_acc, init_mask
from src.vanilla_pytorch.run_model_experiment import run_training
from src.vanilla_pytorch.prune_model import get_masks, update_apply_masks
from src.vanilla_pytorch.prune_model import prune_random
from src.vanilla_pytorch.utils import save_data, plot_graph, init_weights, count_rem_weights
from torch.optim.swa_utils import AveragedModel
from src.vanilla_pytorch.models.convnets import Net2
from src.vanilla_pytorch.models.resnets import Resnets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def update_masks(masks, new_mask):
    """
    Combine the new mask
    :param masks: masks so far
    :param new_mask: new mask from pruned
    """
    for name, mask in masks.items():
        masks[name] = torch.logical_and(mask, new_mask[name])


def handle_og_model(model, args):
    """
    Run, train, setup and save the baseline for the experiment
    :param model: baseline model, weights inited with glorot gaussian
    :param args: HP to use while training
    :return: the original weights of the network, initial masks
    """
    # get hold of w0
    all_masks = {key: mask.to(device) for key, mask in get_masks(model, prune_amts=init_mask)}

    # Run and train the lenet OG, done in run_model_experiment.py.
    # if rewinding is done, return weight dict of 3rd epoch else, w0
    metrics, full_es, _, original_state_dict = run_training(model, device, args=args)
    # original_state_dict
    # TODO: Kept here for testing. Remove after completion
    # metrics, full_es, _ = {'val_score': 0}, 0, 0
    # Save OG model
    torch.save(model.state_dict(), f"{args.model}_OG.pth")
    return original_state_dict, all_masks, {"val_score": metrics['val_score'] * 100,
                                            "full_es": full_es}


def pruned(model, args):
    """
    Run the LTH experiment
    :param args: arguments from cmd or default values
    :param model: model to train
    :return: dictionary with pruning data
    """
    # train a model to 100% as baseline
    original_state_dict, all_masks, baselines = handle_og_model(model, args)
    prune_data = []
    # init a random model
    in_chan = 1 if args.dataset == 'mnist' else 3
    rando_net = globals()[args.model](in_channels=in_chan)
    rando_net.apply(init_weights)
    # set pruning configs
    prune_amt = {'linear': args.pruning_rate_fc / 100, 'conv': args.pruning_rate_conv / 100, 'last': 0.1}
    print(f"START PRUNING")
    for level in range(args.pruning_levels):
        # Prune and get the new mask.
        with torch.no_grad():
            masks = get_masks(model, prune_amts=prune_amt)
            # create a dict that has the same keys as state dict w/o being linked to model.
            detached = dict([(name, mask.clone().to(device)) for name, mask in masks])
            update_masks(all_masks, detached)
            # Load the OG weights and mask it
            model.load_state_dict(copy.deepcopy(original_state_dict))
            model = update_apply_masks(model, all_masks)
            # prune randomly inited model randomly
            prune_random(rando_net, prune_amts=prune_amt)
            non_zero = count_rem_weights(model)
            print(f"Pruning round {level + 1} Weights remaining {non_zero} and 0% is {100 - non_zero}")
        # TODO: Kept here for testing. Remove after completion
        # last_run, pruned_es, training, _ = {'val_score': 0}, 0, 0
        # rand_run, rand_es, rand_training, _ = {'val_score': 0}, 0, 0
        last_run, pruned_es, training, _ = run_training(model, device, args=args)
        rand_run, rand_es, rand_training, _ = run_training(rando_net, device, args)
        prune_data.append({"rem_weight": non_zero,
                           "val_score": last_run['val_score'] * 100,
                           "rand_init": rand_run['val_score'] * 100,
                           "pruned_es": pruned_es,
                           "rand_es": rand_es,
                           "training_data": training,
                           "random_training": rand_training})
    # metrics
    return baselines, prune_data


if __name__ == '__main__':
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='LTH Experiments')
    parser.add_argument('--model', default='Net2',
                        help='Class name of model to train',
                        type=str, choices=['LeNet', 'Net2', 'LeNet300', 'Resnets'])
    parser.add_argument('--batch-size', type=int, default=60,
                        help='input batch size for training (default: 60)')

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='number of iterations to train (default: 50000)')

    parser.add_argument('--lr', type=float, default=1.2e-3,
                        help='learning rate 1.2e-3')

    parser.add_argument('--pruning-rate-conv', type=int, default=20,
                        help='how much to prune a conv layer. taken as a % (default: 20)')
    parser.add_argument('--pruning-rate-fc', type=int, default=20,
                        help='how much to prune a fully connected layer. taken as a % (default: 20)')
    parser.add_argument('--pruning-levels', type=int, default=1,
                        help='No. of times to prune (default: 3), referred to as levels in paper')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    parser.add_argument('--early-stop',
                        action='store_true', help='Does Early if enabled')
    parser.add_argument('--early-delta', type=float, default=0.0005,
                        help='Difference b/w best and current to decide to stop early')
    parser.add_argument('--use-swa',
                        action='store_true', help='Uses SWA if enabled')
    parser.add_argument('--rewind',
                        action='store_true', help='Uses rewining weights to a certain epoch. 1/3')

    parser.add_argument('--name', default='prune',
                        help='name to save data files and plots',
                        type=str)
    # prune to 30 to get 0.1% weights but 25 is ok too
    args = parser.parse_args()

    in_chan, img = (1, 28) if args.dataset == 'mnist' else (3, 32)
    net = globals()[args.model](in_channels=in_chan).to(device)
    net.apply(init_weights)
    summary(net, (in_chan, img, img),
            device=device.type)
    print(f"Arguments: {args}")
    run_data, pruned_data = pruned(net, args)
    run_data["prune_data"] = pruned_data
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    file_name = f"{args.name}_{args.model}_SWA{args.use_swa}_Epochs{args.epochs}_Pruned{args.pruning_levels}"
    stored_at = save_data(run_data, file_name + ".json")
    plot = default_plot_es
    plot['title'] = file_name
    plot['y_max'] = args.epochs
    plot_graph(run_data, plot, file_at=file_name + "_es.png")
    plot = default_plot_acc
    plot['title'] = file_name
    plot['y_max'] = 100
    plot_graph(run_data, plot, file_at=file_name + ".png")
    print(f"Files saved at {stored_at}")
