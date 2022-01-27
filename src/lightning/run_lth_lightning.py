import argparse
import time
from src.lightning.data_lightning import LightningCIFAR10
import pytorch_lightning as pl
from src.lightning.BaseImplementations.BaseModels import count_rem_weights, Net2, init_weights
from src.lightning.BaseImplementations.BaseTrainerAndCallbacks import Pruner, TrainFullModel, RandomPruner
import torch
from os import path, makedirs
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger


# Randomly Init
# Repeat
# Train for j
# prune p% get mask m
# reset remaining to original untrained weights-> reset and apply mask.


# Logging
# Need Final test of each level.  Sparsity vs test score.
# Trainer stores the numbers.

def run_lth_exp(args):

    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)
    # directory for own summary writer to log test acc vs sparsity.
    trial_dir = path.join(args.exp_dir, f"{args.trial}_{args.model}_{args.pruning_levels}")
    logger = SummaryWriter(f"{trial_dir}/tensorboard")
    print(f"Tensorboard logs kept in {logger.log_dir}")

    dm = LightningCIFAR10(batch_size=args.batch_size)
    module = eval(args.model)(learning_rate=args.lr, exp_folder=trial_dir)
    # module = ResNets(learning_rate=args.lr)
    module.apply(init_weights)

    # load from old checkpoint for faster
    # old model that was trained. using it's init weights.
    # module = module.load_from_checkpoint(checkpoint_path="full_trained.ckpt")
    # Logger to track each run.
    logger_path = path.join(trial_dir, f"lightning_tensorboard")
    full_logger = TensorBoardLogger(logger_path, name="full_model_run", version='full_model')
    full_trainer = pl.Trainer(gpus=args.gpu,
                              max_epochs=args.epochs,
                              logger=full_logger,
                              log_every_n_steps=1,
                              val_check_interval=1,
                              check_val_every_n_epoch=1,

                              callbacks=[TrainFullModel()], )
    full_trainer.fit(module, datamodule=dm)
    full_trainer.test(module, datamodule=dm)
    # print(f"Training Done {full_trainer.logged_metrics['test_epoch']['test_acc_epoch']}")
    # # Do i need to reinit my trainer at each time?? reinit new models??
    prune_amt = {'linear': args.pruning_rate_fc / 100, 'conv': args.pruning_rate_conv / 100, 'last': 0.1}
    prune_trainer = pl.Trainer(gpus=args.gpu,
                               max_epochs=args.epochs,
                               num_sanity_val_steps=1,
                               log_every_n_steps=1,
                               check_val_every_n_epoch=1,

                               callbacks=[Pruner(prune_amt)], )
    # random training. init a model with orig weights, prune random
    # random_module = eval(args.model)(learning_rate=args.lr)
    # random_trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[RandomPruner()])  # random one for comparisons
    print(f"PRUNE LOOP")
    for level in range(args.pruning_levels):
        full_logger = TensorBoardLogger(logger_path, name="pruning_runs", version=f"prune_{level + 1}")
        prune_trainer.logger = full_logger
        prune_trainer.fit(module, datamodule=dm)
        prune_trainer.test(module, datamodule=dm)
        test_acc = prune_trainer.logged_metrics['test_epoch']['test_acc_epoch'] * 100
        weight_percent = count_rem_weights(module)
        logger.add_scalar("Sparsity/TestAcc", level * level, weight_percent)
        print(f"End level {level} % test acc {test_acc} weight {weight_percent}")

    print(f"END")


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
    parser.add_argument('--trial', default='Prune',
                        help='name to save data files and plots',
                        type=str)
    parser.add_argument('--exp_dir', type=str, default='experiments', help='path to experiment directory')

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.gpu = 1
    else:
        args.gpu = 0
    run_lth_exp(args)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
