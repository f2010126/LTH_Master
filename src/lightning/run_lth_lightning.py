import argparse
import time
from data_lightning import LightningCIFAR10
import pytorch_lightning as pl
import torch

def run_lth_exp():
    trainer = pl.Trainer(max_epochs=args.epochs)
    # init weights and save weights
    # run model
    # in loop,
    # run model
    # prune,
    # reinit run
    pass

def run_model(args, dm, trainer):
    # run the model as is for n epochs
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model,datamodule=dm)

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

    parser.add_argument('--pruning-rate-conv', type=int, default=20,
                        help='how much to prune a conv layer. taken as a % (default: 20)')
    parser.add_argument('--pruning-rate-fc', type=int, default=20,
                        help='how much to prune a fully connected layer. taken as a % (default: 20)')
    parser.add_argument('--pruning-levels', type=int, default=1,
                        help='No. of times to prune (default: 3), referred to as levels in paper')

    args = parser.parse_args()
    dm = LightningCIFAR10(batch_size=args.batch_size)
    model = eval(args.model)(learning_rate=args.lr)