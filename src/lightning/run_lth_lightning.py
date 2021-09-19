import argparse
import time
from data_lightning import LightningCIFAR10
import pytorch_lightning as pl
import torch
from models import Net2, count_rem_weights
import copy
from pytorch_lightning.callbacks import Callback
from prune_model import get_masks,update_masks,update_apply_masks


# Randomly Init
# Repeat
# Train for j
# prune p% get mask m
# reset remaining to original-> reset and apply mask.

# loop reinit trainer?
# init weights X and save weights X
# fit model X, test Model X
# print/log whateber
# in loop,
# prune model Trainer does it. Model Pruning won't really help
# prune, self.parameters has the weights.
# reinit X
# fit and test X


# Trainer automates the process so it should do things like freeze weights, reinit, checkpoint.
# Let  the model just do regular training
class TrainFullModel(Callback):
    def on_fit_end(self, trainer, pl_module):
        # save trained model here
        trainer.save_checkpoint("full_trained.ckpt")



class PrunerModel(Callback):
    def __init__(self, prune_amt):
        super().__init__()
        self.prune_amt = prune_amt

    def on_fit_start(self, trainer, pl_module):
        # pruning happens here.
        masks = get_masks(pl_module, prune_amts=self.prune_amt)
        # save masks
        detached = dict([(name, mask.clone()) for name, mask in masks])
        update_masks(pl_module.all_masks, detached)
        # reinit old
        checkpoint = torch.load("init_weights.ckpt")  # works
        pl_module.load_state_dict(copy.deepcopy(pl_module.original_state_dict))
        pl_module = update_apply_masks(pl_module, pl_module.all_masks)



    def on_after_backward(self, trainer, pl_module):

        for module in pl_module.modules():
            if hasattr(module, "weight_mask"):
                weight = next(param for name, param in module.named_parameters() if "weight" in name)
                weight.grad = weight.grad * module.weight_mask



class RandomPruner(Callback):
    def on_fit_start(self, trainer, pl_module):
        print(f"start at the begining re init weights")
        pl_module.load_from_checkpoint(checkpoint_path="init_weights.ckpt")

    def on_after_backward(self, trainer, pl_module):
        print(f"Freeze weights here")
        pass

    def on_test_end(self, trainer, pl_module):
        print(f"sparity {count_rem_weights(pl_module)}")



def run_lth_exp(args):
    dm = LightningCIFAR10(batch_size=args.batch_size)
    module = eval(args.model)(learning_rate=args.lr)

    full_trainer = pl.Trainer(max_epochs=args.epochs,
                              default_root_dir='lth_loggers/',
                              num_sanity_val_steps=0,
                              limit_val_batches=1,
                              limit_train_batches=1,
                              limit_test_batches=1,
                              log_every_n_steps=5,
                              callbacks=[TrainFullModel()], )
    full_trainer.fit(module, datamodule=dm)
    full_trainer.test(module, datamodule=dm)

    # # Do i need to reinit my trainer at each time?? reinit new models??
    prune_amt = {'linear': args.pruning_rate_fc / 100, 'conv': args.pruning_rate_conv / 100, 'last': 0.1}
    prune_trainer = pl.Trainer(max_epochs=args.epochs,
                               num_sanity_val_steps=0,
                               limit_val_batches=1,
                               limit_train_batches=1,
                               limit_test_batches=1,
                               log_every_n_steps=5,
                               callbacks=[PrunerModel(prune_amt)],
                               default_root_dir='prune_loggers/')
    # random_trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[RandomPruner()])  # random one for comparisons
    print(f"PRUNE LOOP")
    args.pruning_levels = 5
    for level in range(args.pruning_levels):
        print(f"Start level {level} % {count_rem_weights(module)}")
        prune_trainer.fit(module, datamodule=dm)
        prune_trainer.test(module, datamodule=dm)
        print(f"End level {level} % {count_rem_weights(module)}")

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

    args = parser.parse_args()
    run_lth_exp(args)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
