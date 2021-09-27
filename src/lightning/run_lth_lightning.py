import argparse
import time
from data_lightning import LightningCIFAR10
import pytorch_lightning as pl
from BaseImplementations.BaseModels import count_rem_weights,Net2
from BaseImplementations.BaseTrainerAndCallbacks import Pruner,TrainFullModel, RandomPruner
import torch



# Randomly Init
# Repeat
# Train for j
# prune p% get mask m
# reset remaining to original untrained weights-> reset and apply mask.


# Logging
# Need Final test of each level.  Sparsity vs test score.
# Trainer stores the numbers.

def run_lth_exp(args):
    args.epochs = 3
    dm = LightningCIFAR10(batch_size=args.batch_size)
    module = eval(args.model)(learning_rate=args.lr)
    # load from old checkpoint for faster
    # old model that was trained. using it's init weights.
    # module = module.load_from_checkpoint(checkpoint_path="full_trained.ckpt")
    logger_name = f"{args.name}_{args.model}_{args.pruning_levels}/"

    full_trainer = pl.Trainer(max_epochs=args.epochs,
                              default_root_dir=logger_name,
                              num_sanity_val_steps=0,
                              log_every_n_steps=5,
                              limit_train_batches=10,
                              limit_val_batches=10,
                              limit_test_batches=10,
                              check_val_every_n_epoch=1,
                              callbacks=[TrainFullModel()], )
    full_trainer.fit(module, datamodule=dm)
    full_trainer.test(module, datamodule=dm)

    # # Do i need to reinit my trainer at each time?? reinit new models??
    prune_amt = {'linear': args.pruning_rate_fc / 100, 'conv': args.pruning_rate_conv / 100, 'last': 0.1}
    prune_trainer = pl.Trainer(max_epochs=args.epochs,
                               num_sanity_val_steps=0,
                               log_every_n_steps=5,
                               limit_train_batches=10,
                               limit_val_batches=10,
                               limit_test_batches=10,
                               check_val_every_n_epoch=1,
                               callbacks=[Pruner(prune_amt)],
                               default_root_dir=logger_name)
    # random training. init a model with orig weights, prune random
    # random_module = eval(args.model)(learning_rate=args.lr)
    # random_trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[RandomPruner()])  # random one for comparisons
    print(f"PRUNE LOOP")
    args.pruning_levels = 2
    for level in range(args.pruning_levels):
        print(f"Start level {level} % {count_rem_weights(module)}  conv1 has {torch.count_nonzero(module.conv1.weight)} Zeros")
        prune_trainer.fit(module, datamodule=dm)
        prune_trainer.test(module, datamodule=dm)
        print(f"End level {level} % test acc {full_trainer.logged_metrics['test_epoch']['test_acc_epoch'] * 100} weight % {count_rem_weights(module)}")

    print(f"END")


def run_model(datamod, module, trainer):
    # run the model as is for n epochs
    trainer.fit(module, datamodule=datamod)
    trainer.test(module, datamodule=datamod)


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
    parser.add_argument('--name', default='Prune',
                        help='name to save data files and plots',
                        type=str)

    args = parser.parse_args()
    run_lth_exp(args)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
