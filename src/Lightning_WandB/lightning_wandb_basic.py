import os
import yaml
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

try:
    from BaseLightningModule.base_module import LitSystem94Base
    from BaseLightningModule.data_module import Custom_CIFAR10DataModule
    from utils import checkdir, get_data_module
    from config import AttrDict

except ImportError:
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystem94Base
    from src.Lightning_WandB.BaseLightningModule.data_module import Custom_CIFAR10DataModule
    from src.Lightning_WandB.utils import checkdir, get_data_module
    from src.Lightning_WandB.BaseLightningModule.base_module import LitSystem94Base
    from src.Lightning_WandB.config import AttrDict


def execute_trainer(args=None):
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

    PATH_DATASETS = args.data_root
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 256 if AVAIL_GPUS else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    cifar10_module = get_data_module(path=PATH_DATASETS, batch=BATCH_SIZE,
                                     seed=args.seed, workers=NUM_WORKERS)
    # cifar10_module=Custom_CIFAR10DataModule()
    model = LitSystem94Base(batch_size=BATCH_SIZE, arch=args.model, lr=0.05)
    model.datamodule = cifar10_module

    exp_dir = os.path.join(os.getcwd(), args.exp_dir)
    checkdir(exp_dir)

    trial_dir = path.join(exp_dir, args.trial)
    checkdir(exp_dir)
    print(f"All Saved logs at {trial_dir}")
    checkdir(f"{trial_dir}/wandb_logs")

    wandb_logger = WandbLogger(project=args.wand_exp_name, job_type='train',
                               save_dir=f"{trial_dir}/wandb_logs", config=args,
                               name=args.trial, )
    # early_stop_callback = EarlyStopping('val_loss',min_delta=0.03, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode="max",
        dirpath=f"{trial_dir}/models",
        filename='sample-cifar10-{epoch:02d}-{val_acc:.2f}',
        verbose=True)

    trainer = Trainer(
        gpus=args.gpus, num_nodes=args.nodes, accelerator="ddp",
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback],
        checkpoint_callback=True
    )

    trainer.fit(model, cifar10_module)
    trainer.test(model, datamodule=cifar10_module)
    wandb.finish()


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Class name of model to train  (default: resnet18)',
                        choices=['resnet18', 'torch_resnet'])
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate  (default: 1.2e-3)')
    parser.add_argument('--batch-size', type=int, default=60,
                        help='input batch size for training (default: 60)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training  (default: cifar10 )')
    parser.add_argument('--seed', default=123, type=int, metavar='N',
                        help='random seed of numpy and torch  (default: 123)')
    parser.add_argument('--data_root', type=str, default='data',
                        help='path to dataset directory (default: data)')
    parser.add_argument('--exp_dir', type=str, default='experiments',
                        help='path to experiment directory(default: experiments)')
    parser.add_argument('--wand_exp_name', type=str, default='wandb-lightning_Single',
                        help='Name the project for wandb (default: wandb-lightning_Single)')
    parser.add_argument('--trial', type=str, default='1', help='trial id (default: 1)')
    parser.add_argument('--early-stop',
                        action='store_true', help='Uses Early Stop if enabled (default: False)')
    parser.add_argument('--config_file_name', type=str, default='basic_lightning.yaml',
                        help='Name of config file (default:basic_lightning.yaml)')
    parser.add_argument('--gpus', default=1, type=int, metavar='G', help='# of GPUs (default: 1)')
    parser.add_argument('--nodes', default=1, type=int, metavar='O', help='# of nodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=30000,
                        help='# iterations to train (default: 30k)')

    args = parser.parse_args()

    # Load config path then args
    config_path = os.path.join(os.getcwd(), "src/configs")
    print(f"Config path --> {config_path}")
    with open(f"{config_path}/{args.config_file_name}", "r") as f:
        config = yaml.safe_load(f)
    config["config_file_name"] = args.config_file_name
    config = AttrDict(config)
    print(config)
    execute_trainer(config)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
