try:
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
    from BaseLightningModule.base_module import LitSystem
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

except ImportError:
    from BaseLightningModule.base_module import LitSystem
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
    from BaseLightningModule.base_module import LitSystem
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_data_module(path, batch, workers=0):
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ])
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=path,
        batch_size=batch,
        num_workers=workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    return cifar10_dm


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

    cifar10_module = get_data_module(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    model = LitSystem(batch_size=BATCH_SIZE, arch="resnet18", lr=0.05)
    model.datamodule = cifar10_module

    exp_dir = os.path.join(os.getcwd(), args.exp_dir)
    if not path.exists(exp_dir):
        makedirs(exp_dir)

    trial_dir = path.join(exp_dir, args.trial)
    print(f"Saved logs at {trial_dir}")
    wandb_logger = WandbLogger(project='wandb-lightning_Single', job_type='train', save_dir=f"{trial_dir}/wandb_logs")
    logger = TensorBoardLogger("lightning_logs/", name="resnet")
    early_stop_callback = EarlyStopping('val_loss')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"{trial_dir}/models",
        filename='sample-cifar10-{epoch:02d}-{val_loss:.2f}')

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        gpus=AVAIL_GPUS,
        logger=wandb_logger,
        callbacks=[early_stop_callback,
                   LearningRateMonitor(logging_interval="step"),
                   checkpoint_callback],
        checkpoint_callback=True
    )

    trainer.fit(model, cifar10_module)
    trainer.test(model, datamodule=cifar10_module)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Class name of model to train',
                        choices=['resnet18', 'Net2', 'LeNet300', 'Resnets'])
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
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    args = parser.parse_args()

    execute_trainer(args)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
