# use the data augmentation from the other repo
# do a normal LTH workflow
# add the same trainng workflow from repo
# prune
# Don't prune the Last layer.
# KNN validation?

import yaml
import os
import wandb
from os import path
import torch
import argparse
import time
import warnings
import torch.backends.cudnn as cudnn
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

import copy

try:
    from BaseLightningModule.base_module import LitSystemPrune,LitSystemSSLPrune
    from utils import checkdir, get_data_module, layer_looper, apply_prune, \
        reset_weights, count_rem_weights, set_experiment_run, add_callbacks
    from BaseLightningModule.callbacks import FullTrainer, PruneTrainer
    from config import AttrDict
except ImportError:
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystem94Base,LitSystemSSLPrune
    from src.Lightning_WandB.utils import checkdir, get_data_module, \
        layer_looper, apply_prune, reset_weights, count_rem_weights, set_experiment_run, add_callbacks
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystemPrune, LitSystemRandom
    from src.Lightning_WandB.BaseLightningModule.callbacks import FullTrainer, PruneTrainer
    from src.Lightning_WandB.config import AttrDict


def execute_trainer(args):
    # loop the trainer n times and log each run separately under exeperiment/wand_exp_name/
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    NUM_WORKERS = int(os.cpu_count() / 2)

    trial_dir = set_experiment_run(args)
    print(f"All Saved logs at {trial_dir}/wandb_logs All Models in {trial_dir}/models")
    checkdir(f"{trial_dir}/wandb_logs")

    ### BASELINE###
    cifar10_module = get_data_module(path=args.data_root, batch=args.batch_size,
                                     seed=args.seed, workers=NUM_WORKERS)

    model = LitSystemSSLPrune(batch_size=args.batch_size, experiment_dir=f"{trial_dir}/models/baseline", arch=args.model,
                           lr=args.learning_rate, reset_itr=args.reset_itr)
    model.datamodule = cifar10_module

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='ssl_resnet',
                        help='Class name of model to train',
                        choices=['resnet18', 'torch_resnet', 'ssl_resnet'])
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='number of iterations to train (default: 50000)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate 1.2e-3')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed of numpy and torch')
    parser.add_argument('--data_root', type=str, default='data', help='path to dataset directory')
    parser.add_argument('--exp_dir', type=str, default='experiments', help='path to experiment directory')
    parser.add_argument('--wand_exp_name', type=str, default='Looper',
                        help='Name the project for wandb')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--levels', type=int, default=1,
                        help='Prune Levels (default: 1)')
    parser.add_argument('--early-stop',
                        action='store_true', help='Uses Early Stop if enabled')
    parser.add_argument('--config_file_name', type=str, default='4_15_lth_ssl.yaml', help='Name of config file')
    parser.add_argument('--reset_itr', type=int, default=500,
                        help='epoch reset weights to (default: 0)')
    parser.add_argument('--gpus', default=1, type=int, metavar='G', help='# of GPUs')
    parser.add_argument('--nodes', default=1, type=int, metavar='O', help='# of nodes')
    parser.add_argument('--swa',
                        action='store_true', help='Uses SWA as part of optimiser if enabled')
    parser.add_argument('--prune_global',
                        action='store_false', help='Prune Layer wise or globally. Default global')
    parser.add_argument('--val_freq', default=1, type=int, metavar='O', help='frequency of validation')
    parser.add_argument('--es_patience', default=5, type=int, metavar='O', help='when to Early stop')
    parser.add_argument('--es_delta', type=float, default=0.01,
                        help='delta for early stopping')
    parser.add_argument('--pruning_amt', type=float, default=0.2,
                        help='how much to prune a conv layer. (default: 0.2)')

    args = parser.parse_args()
    # Load config path then args
    config_path = os.path.join(os.getcwd(), "src/configs")

    # "/Users/diptisengupa/Desktop/CODEWORK/GitHub/SS2021/LTH_Project/ReproducingResults/LTH_Master/src" \
    #               "/configs"
    with open(f"{config_path}/{args.config_file_name}", "r") as f:
        config = yaml.safe_load(f)

    config["gpus"] = args.epochs
    config = AttrDict(config)
    # override config values
    execute_trainer(config)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
