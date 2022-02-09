try:
    import os
    import wandb
    from os import path, makedirs
    import torch
    import argparse
    import time
    import warnings
    import torch.backends.cudnn as cudnn
    from pytorch_lightning import seed_everything, Trainer
    from pl_bolts.datamodules import CIFAR10DataModule
    import torchvision
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
    from BaseLightningModule.base_module import LitSystemPrune
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from utils import checkdir, get_data_module

except ImportError:
    import wandb
    import os
    from os import path, makedirs
    import torch
    import argparse
    import time
    import warnings
    import torch.backends.cudnn as cudnn
    from pytorch_lightning import seed_everything, Trainer
    from pl_bolts.datamodules import CIFAR10DataModule
    import torchvision
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
    from .BaseLightningModule.base_module import LitSystem94Base
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from .utils import checkdir, get_data_module
    from .BaseLightningModule.base_module import LitSystemPrune


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

    PATH_DATASETS = args.data_root
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 256 if AVAIL_GPUS else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    cifar10_module = get_data_module(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    model = LitSystemPrune(batch_size=args.batch_size, arch=args.model, lr=args.lr)
    model.datamodule = cifar10_module

    exp_dir = os.path.join(os.getcwd(), args.exp_dir)
    checkdir(exp_dir)

    trial_dir = path.join(exp_dir, args.trial)
    checkdir(exp_dir)
    print(f"All Saved logs at {trial_dir}")
    checkdir(f"{trial_dir}/wandb_logs")
    wandb.init(config=args, project=args.wand_exp_name,
               job_type='train', dir=f"{trial_dir}/wandb_logs", group='MultipleRuns', name="10Shot")

    # do wandb.define_metric() after wandb.init()
    # Define the custom x axis metric
    wandb.define_metric("weight_pruned")
    # Define which metrics to plot against that x-axis
    wandb.define_metric("pruned-test-acc", step_metric='weight_pruned')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode="max",
        dirpath=f"{trial_dir}/models",
        filename='sample-cifar10-{epoch:02d}-{val_acc:.2f}',
        verbose=True)

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        limit_train_batches=20,
        limit_val_batches=20,
        limit_test_batches=20,
        gpus=AVAIL_GPUS,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback],
        enable_checkpointing=True
    )

    for i in range(1):
        # log Test Acc vs weight %
        trainer.fit(model, cifar10_module)
        trainer.test(model, datamodule=cifar10_module)
        test_acc = i * i
        weight_prune = 100 - i
        print(f"Weight % here {weight_prune}")
        wandb.log({"pruned-test-acc": test_acc, "weight_pruned": weight_prune})


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Class name of model to train',
                        choices=['resnet18', 'torch_resnet', 'LeNet300', 'Resnets'])
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='number of iterations to train (default: 50000)')
    parser.add_argument('--lr', type=float, default=1.2e-3,
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

    args = parser.parse_args()

    execute_trainer(args)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
