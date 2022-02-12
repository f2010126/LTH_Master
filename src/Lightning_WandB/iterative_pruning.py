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
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import copy

try:
    from BaseLightningModule.base_module import LitSystemPrune
    from utils import checkdir, get_data_module, layer_looper, apply_pruning, \
        reset_weights, count_rem_weights, check_model_change
    from config import AttrDict
    from BaseLightningModule.callbacks import FullTrainer, PruneTrainer
except ImportError:
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystem94Base
    from src.Lightning_WandB.utils import checkdir, get_data_module, \
        layer_looper, apply_pruning, reset_weights, count_rem_weights, check_model_change
    from src.Lightning_WandB.config import AttrDict
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystemPrune
    from src.Lightning_WandB.BaseLightningModule.callbacks import FullTrainer, PruneTrainer


def set_experiment_run(args):
    exp_dir = os.path.join(os.getcwd(), args.exp_dir)
    checkdir(exp_dir)
    trial_dir = path.join(exp_dir, args.trial)
    checkdir(exp_dir)
    return trial_dir


def execute_trainer(args):
    # loop the trainer n times and log each run separately under exeperiment/trial/log/{run_#}
    if args.seed is not None:
        seed_everything(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 256 if AVAIL_GPUS else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    trial_dir = set_experiment_run(args)
    print(f"All Saved logs at {trial_dir}")
    checkdir(f"{trial_dir}/wandb_logs")

    cifar10_module = get_data_module(args.data_root, args.batch_size, NUM_WORKERS)
    model = LitSystemPrune(batch_size=args.batch_size, experiment_dir=f"{trial_dir}/models", arch=args.model,
                           lr=args.learning_rate)
    model.datamodule = cifar10_module
    print(f"Keys OG \n {model.original_wgts.keys()}")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode="max",
        dirpath=f"{trial_dir}/models",
        filename='resnet-cifar10-{epoch:02d}-{val_acc:.2f}',
        save_last=True,
        verbose=True)

    # BASELINE RUN
    wandb_logger = WandbLogger(project=args.wand_exp_name, save_dir=f"{trial_dir}/wandb_logs",
                               reinit=True, config=args, job_type='initial-baseline',
                               group=args.trial, name=f"baseline_run")
    full_trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        gpus=AVAIL_GPUS,
        callbacks=[FullTrainer(),
                   checkpoint_callback],
        checkpoint_callback=True,
        logger=wandb_logger
    )

    full_trainer.fit(model, cifar10_module)
    full_trainer.test(model, datamodule=cifar10_module)
    test_acc = full_trainer.logged_metrics['test_acc'] * 100
    print(f"Test Acc {test_acc}")
    weight_prune = count_rem_weights(model)
    print(f"BaseLine Weight % {weight_prune}")
    wandb.define_metric("weight_pruned")
    wandb.define_metric("pruned-test-acc", step_metric='weight_pruned')
    wandb.log({"pruned-test-acc": test_acc * 100, 'weight_pruned': weight_prune})
    wandb.finish()

    full_cap = copy.deepcopy(model.state_dict())

    # PRUNING LOOP
    for i in range(args.levels):
        # log Test Acc vs weight %
        # PRUNE
        apply_pruning(model, 0.2)
        # RESET TO SAVED WEIGHTS
        reset_weights(model, model.original_wgts)
        weight_prune = count_rem_weights(model)
        print(f" PRUNING LEVEL #{i+1} Weight % {weight_prune}")

        # RETRAIN
        # Reinit the Trainer.
        prune_trainer = Trainer(
            progress_bar_refresh_rate=10,
            max_epochs=args.epochs,
            gpus=AVAIL_GPUS,
            callbacks=[
                checkpoint_callback],
            checkpoint_callback=True,
            logger=wandb_logger
        )
        wandb_logger = WandbLogger(project=args.wand_exp_name, save_dir=f"{trial_dir}/wandb_logs",
                                   reinit=True, config=args, job_type=f'pruning_level_{weight_prune}',
                                   group=args.trial, name=f"run_#_{i}")
        prune_trainer.fit(model, cifar10_module)
        prune_trainer.test(model, datamodule=cifar10_module)
        test_acc = prune_trainer.logged_metrics['test_acc'] * 100
        print(f"Test Acc {test_acc}")
        # do wandb.define_metric() after wandb.init()
        # Define the custom x axis metric, and define which metrics to plot against that x-axis
        wandb.define_metric("weight_pruned")
        wandb.define_metric("pruned-test-acc", step_metric='weight_pruned')
        wandb.log({"pruned-test-acc": test_acc * 100, 'weight_pruned': weight_prune})
        wandb.finish()
        full_cap = copy.deepcopy(model.state_dict())


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Class name of model to train',
                        choices=['resnet18', 'torch_resnet'])
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='number of iterations to train (default: 50000)')
    parser.add_argument('--learning_rate', type=float, default=1.2e-3,
                        help='learning rate 1.2e-3')
    parser.add_argument('--batch-size', type=int, default=60,
                        help='input batch size for training (default: 60)')
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
    parser.add_argument('--config_file_name', type=str, default='default_lth_reset.yaml', help='Name of config file')
    parser.add_argument('--reset_epoch', type=int, default=0,
                        help='epoch reset weights to (default: 0)')
    args = parser.parse_args()
    # Load config path then args
    config_path = os.path.join(os.getcwd(), "src/configs")

    # "/Users/diptisengupa/Desktop/CODEWORK/GitHub/SS2021/LTH_Project/ReproducingResults/LTH_Master/src" \
    #               "/configs"
    with open(f"{config_path}/{args.config_file_name}", "r") as f:
        config = yaml.safe_load(f)

    config["epochs"] = args.epochs
    config["trial"] = args.trial
    config["levels"] = args.levels
    config["reset_epoch"] = args.reset_epoch
    config = AttrDict(config)
    # override config values
    execute_trainer(config)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
