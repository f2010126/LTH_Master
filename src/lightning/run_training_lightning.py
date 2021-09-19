import argparse
import time
import torch
from models import Net2, count_rem_weights
from data_lightning import LightningCIFAR10
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelPruning
import os


class TrainerCallbacks(Callback):

    def on_fit_end(self, trainer, pl_module):
        print(f"Normal trainer. fit end check weights here {count_rem_weights(pl_module)}")

    def on_test_end(self, trainer, pl_module):
        print(f"Normal trainer.test end check weights here {count_rem_weights(pl_module)}")


def run_training(args, model, dm):
    trainer = pl.Trainer(max_epochs=args.epochs,
                         weights_summary="full",
                         default_root_dir='test_loggers/',
                         log_every_n_steps=1,
                         profiler="simple",
                         fast_dev_run=True,
                         callbacks=[TrainerCallbacks()])
    trainer.fit(model, datamodule=dm)
    print(f"TEST")
    trainer.test(model=model, datamodule=dm)
    pass

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='Net2',
                        help='Class name of model to train',
                        choices=['LeNet', 'Net2', 'LeNet300', 'Resnets'])
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

    # Training settings
    args = parser.parse_args()
    loss = torch.nn.CrossEntropyLoss()
    dm = LightningCIFAR10(batch_size=args.batch_size)
    loss = torch.nn.CrossEntropyLoss()
    model = eval(args.model)(learning_rate=args.lr)  # Net2()
    args.epochs = 2

    run_training(args,model,dm)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
