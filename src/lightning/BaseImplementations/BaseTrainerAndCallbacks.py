from pytorch_lightning import Trainer
from src.lightning.prune_model import get_masks, update_apply_masks
import torch
from pytorch_lightning.callbacks import Callback
from src.lightning.BaseImplementations.BaseModels import count_rem_weights


class BaseTrainerCallbacks(Callback):
    pass


# Trainer automates the process, so it should do things like freeze weights, reinit, checkpoint.
# Let  the model do regular training
class TrainFullModel(Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.save_checkpoint("init_trainer_weights.ckpt")

    def on_fit_end(self, trainer, pl_module):
        # save trained model here
        trainer.save_checkpoint("full_trained.ckpt")


class Pruner(Callback):
    def __init__(self, prune_amt):
        super().__init__()
        self.prune_amt = prune_amt

    def on_fit_start(self, trainer, pl_module):
        # pruning happens here.
        masks = get_masks(pl_module, prune_amts=self.prune_amt)
        # reinit old
        checkpt = torch.load("init_trainer_weights.ckpt")
        pl_module.load_state_dict(checkpt['state_dict'])
        pl_module = update_apply_masks(pl_module, masks)
        print(f"Masks updated? :( {pl_module.conv1.weight[0][0]} {count_rem_weights(pl_module)}")

    def on_after_backward(self, trainer, pl_module):
        pass
        # for module in pl_module.children():
        #     mask = module.weight_mask
        #     weight = list(module.named_parameters())[1][1]
        # grad already reduced but not weights
        # if hasattr(module, "weight_mask"):
        #     weight = next(param for name, param in module.named_parameters() if "weight" in name)
        #     weight.grad = weight.grad * module.weight_mask


class RandomPruner(Callback):
    def on_fit_start(self, trainer, pl_module):
        pl_module.load_from_checkpoint(checkpoint_path="init_weights.ckpt")

    def on_after_backward(self, trainer, pl_module):
        print(f"Freeze weights here")
        pass

    def on_test_end(self, trainer, pl_module):
        print(f"sparity {count_rem_weights(pl_module)}")


# Logging
# Need Final test of each level.  Sparsity vs test score.
# Trainer stores the numbers.

class BaseTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.masks = None
