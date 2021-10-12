import argparse
import time
import torch
import pytorch_lightning as pl
from data_lightning import LightningCIFAR10
from src.lightning.BaseImplementations.BaseModels import Net2, count_rem_weights, init_weights
from src.lightning.BaseImplementations.BaseTrainerAndCallbacks import BaseTrainerCallbacks, BaseTrainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging


def run_training(args):
    if torch.cuda.is_available():
        args.gpu = 1
    else:
        args.gpu = 0
    args.epochs = 3
    args.early_stop = False
    args.use_swa = False

    dm = LightningCIFAR10(batch_size=args.batch_size)
    model = eval(args.model)(learning_rate=args.lr)  # Net2()
    model.apply(init_weights)
    logger_name = f"{args.name}_{args.model}_{args.use_swa}/"
    trainer_callbacks = [BaseTrainerCallbacks()]
    if args.early_stop:
        trainer_callbacks.append(
            EarlyStopping(monitor="val_loss_epoch", min_delta=0.1, patience=2, verbose=True, mode="min"))
    trainer = pl.Trainer(gpus=args.gpu,
                         max_epochs=args.epochs,
                         stochastic_weight_avg=args.use_swa,
                         val_check_interval=1,
                         default_root_dir=f'{args.name}_loggers/',
                         check_val_every_n_epoch=1,
                         log_every_n_steps=1,
                         callbacks=trainer_callbacks)
    trainer.fit(model, datamodule=dm)
    print(f"TEST")
    # trainer.logged_metrics
    # all metrics of Last step/epoch But care about only test metrics
    trainer.test(model=model, datamodule=dm)
    print(f"Logged {trainer.logged_metrics} with weight % {count_rem_weights(model)}")
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
    parser.add_argument('--use-swa',
                        action='store_true', help='Uses SWA if enabled')
    parser.add_argument('--early-stop',
                        action='store_true', help='Does Early if enabled')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    parser.add_argument('--name', default='Exp_swa_yes',
                        help='name to save data files and plots',
                        type=str)

    # Training settings
    args = parser.parse_args()
    run_training(args)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
