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
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import copy

try:
    from BaseLightningModule.base_module import LitSystemPrune
    from utils import checkdir, get_data_module, layer_looper, apply_prune, \
        reset_weights, count_rem_weights, remove_pruning
    from BaseLightningModule.callbacks import FullTrainer, PruneTrainer
    from config import AttrDict
except ImportError:
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystem94Base
    from src.Lightning_WandB.utils import checkdir, get_data_module, \
        layer_looper, apply_prune, reset_weights, count_rem_weights, remove_pruning
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystemPrune, LitSystemRandom
    from src.Lightning_WandB.BaseLightningModule.callbacks import FullTrainer, PruneTrainer
    from src.Lightning_WandB.config import AttrDict


def set_experiment_run(args):
    exp_dir = os.path.join(os.getcwd(), args.exp_dir)
    checkdir(exp_dir)
    trial_dir = path.join(exp_dir, args.wand_exp_name)
    checkdir(exp_dir)
    return trial_dir


def add_callbacks(args):
    call_list = [LearningRateMonitor(logging_interval="step"),
                 TQDMProgressBar(refresh_rate=100)]
    if args.early_stop:
        early_stopping = EarlyStopping('val_loss', patience=args.es_patience, mode='min', min_delta=args.es_delta,
                                       verbose=True)
        call_list.append(early_stopping)

    return call_list


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

    # AVAIL_GPUS = min(1, torch.cuda.device_count())
    # BATCH_SIZE = 256 if AVAIL_GPUS else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    trial_dir = set_experiment_run(args)
    print(f"All Saved logs at {trial_dir}/wandb_logs All Models in {trial_dir}/models")
    checkdir(f"{trial_dir}/wandb_logs")

    ### BASELINE###
    cifar10_module = get_data_module(path=args.data_root, batch=args.batch_size,
                                     seed=args.seed, workers=NUM_WORKERS)

    model = LitSystemPrune(batch_size=args.batch_size, experiment_dir=f"{trial_dir}/models/baseline", arch=args.model,
                           lr=args.learning_rate, reset_itr=args.reset_itr)
    model.datamodule = cifar10_module

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode="max",
        dirpath=f"{trial_dir}/models/baseline",
        filename='resnet-cifar10-{epoch:02d}-{val_acc:.2f}',
        save_last=False)
    callback_list = add_callbacks(args)
    callback_list.extend([checkpoint_callback, FullTrainer()])

    wandb_logger = WandbLogger(project=args.wand_exp_name, save_dir=f"{trial_dir}/wandb_logs/baseline",
                               reinit=True, config=args, job_type='initial-baseline',
                               group='Baseline', name=f"baseline_run")

    full_trainer = Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        gpus=-1, num_nodes=1, strategy='ddp',
        callbacks=callback_list,
        stochastic_weight_avg=args.swa_enabled,
        enable_checkpointing=True,
        logger=wandb_logger,
        deterministic=True,
        check_val_every_n_epoch=args.val_freq,
    )

    full_trainer.fit(model, cifar10_module)
    full_trainer.test(model, datamodule=cifar10_module)
    test_acc = full_trainer.logged_metrics['test_acc']
    print(f"BaseLine Weight % {count_rem_weights(model)} with Test Acc {test_acc}")
    wandb.finish()

    model.experiment_dir = f"{trial_dir}/models/pruned"
    weight_cent = 1
    # PRUNING LOOP
    for i in range(args.levels):
        # log Test Acc vs weight %
        # PRUNE L1Unstructured, reset weights
        apply_prune(model, args.pruning_amt, "magnitude", args.prune_global)
        reset_weights(model, model.original_wgts)
        # remove_pruning(model) # make it permanent
        weight_prune = count_rem_weights(model)

        # reinit a random model.
        # initialise a model with random weights and get a randomly pruned model of that sparsity
        randomModel = LitSystemRandom(batch_size=args.batch_size,
                                      experiment_dir=f"{trial_dir}/models/random",
                                      arch=args.model, lr=args.learning_rate)
        randomModel.datamodule = cifar10_module
        weight_cent *= 1 - args.pruning_amt
        apply_prune(randomModel, 1 - weight_cent, "random", args.prune_global)
        remove_pruning(randomModel)

        print(
            f" PRUNING LEVEL #{i + 1} Model weight % here {weight_prune} Random Weight % here {count_rem_weights(randomModel)}")

        wandb_logger = WandbLogger(project=args.wand_exp_name, save_dir=f"{trial_dir}/wandb_logs/pruned",
                                   reinit=True, config=args, job_type=f'level_{weight_prune}',
                                   group='Pruning', name=f"pruning_#_{i}")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode="max",
            verbose=True,
            dirpath=f"{trial_dir}/models/pruned/level_{i + 1}",
            filename='resnet-pruned-{epoch:02d}-{val_acc:.2f}',
            save_last=False, )
        callback_list = add_callbacks(args)
        callback_list.extend([checkpoint_callback])

        prune_trainer = Trainer(
            max_epochs=args.epochs,
            max_steps=args.max_steps,
            gpus=-1, num_nodes=1, strategy='ddp',
            callbacks=callback_list,
            stochastic_weight_avg=args.swa_enabled,
            enable_checkpointing=True,
            logger=wandb_logger,
            deterministic=True,
            check_val_every_n_epoch=args.val_freq,
        )
        prune_trainer.fit(model, cifar10_module)
        prune_trainer.test(model, datamodule=cifar10_module, ckpt_path='best')
        test_acc = prune_trainer.logged_metrics['test_acc']
        print(
            f"Pruned Test Acc {test_acc} and best model score {checkpoint_callback.best_model_score} at {checkpoint_callback.best_model_path}")
        wandb.finish()

        # Randomly inited Trained
        random_wandb_logger = WandbLogger(project=args.wand_exp_name, save_dir=f"{trial_dir}/wandb_logs/random",
                                          reinit=True, config=args, job_type=f'level_{weight_prune}',
                                          group='Random', name=f"random_#_{i}")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode="max",
            dirpath=f"{trial_dir}/models/random/level_{i + 1}",
            filename='resnet-random-{epoch:02d}-{val_acc:.2f}',
            save_last=False, )
        callback_list = add_callbacks(args)
        callback_list.extend([checkpoint_callback])
        random_trainer = Trainer(
            max_epochs=args.epochs,
            max_steps=args.max_steps,
            gpus=-1, num_nodes=1, strategy='ddp',
            callbacks=callback_list,
            stochastic_weight_avg=args.swa_enabled,
            enable_checkpointing=True,
            logger=random_wandb_logger,
            deterministic=True,
            check_val_every_n_epoch=args.val_freq,
        )
        random_trainer.fit(randomModel, cifar10_module)
        random_trainer.test(randomModel, datamodule=cifar10_module, ckpt_path='best')
        random_test_acc = random_trainer.logged_metrics['test_acc']
        print(
            f"Random Test Acc {random_test_acc} and best model {checkpoint_callback.best_model_score} at {checkpoint_callback.best_model_path}")
        wandb.finish()


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
    parser.add_argument('--config_file_name', type=str, default='4_2_lth_default.yaml', help='Name of config file')
    parser.add_argument('--reset_itr', type=int, default=0,
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
