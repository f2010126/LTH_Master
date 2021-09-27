from pytorch_lightning import Trainer
from src.lightning.prune_model import get_masks, update_masks, update_apply_masks
import copy
from pytorch_lightning.callbacks import Callback
from src.lightning.BaseImplementations.BaseModels import count_rem_weights



class BaseTrainerCallbacks(Callback):
    pass


# Trainer automates the process so it should do things like freeze weights, reinit, checkpoint.
# Let  the model just do regular training
class TrainFullModel(Callback):

    def on_fit_end(self, trainer, pl_module):
        # save trained model here
        print(f"OG Weighst {pl_module.original_state_dict['conv1.weight_orig'][0][0]}")
        print(f"END WEIGHTS {pl_module.conv1.weight[0][0]}")
        trainer.save_checkpoint("full_trained.ckpt")



class Pruner(Callback):
    def __init__(self, prune_amt):
        super().__init__()
        self.prune_amt = prune_amt

    def on_fit_start(self, trainer, pl_module):
        # pruning happens here.
        print(f"PRE PRUNE module conv1 {pl_module.conv1.weight[0][0]} ")
        masks = get_masks(pl_module, prune_amts=self.prune_amt)
        print(
            f"POST PRUNE module conv1 {pl_module.conv1.weight[0][0]} ")

        # save masks
        detached = dict([(name, mask.clone()) for name, mask in masks])
        update_masks(pl_module.all_masks, detached)
        # reinit old
        # checkpoint = torch.load("init_weights.ckpt")  # works
        pl_module.load_state_dict(copy.deepcopy(pl_module.original_state_dict))
        print(f"OG Weighst {pl_module.original_state_dict['conv1.weight_orig'][0][0]}")
        print(
            f"LOADING OG STATE DICT module conv1 {pl_module.conv1.weight[0][0]} ")

        pl_module = update_apply_masks(pl_module, pl_module.all_masks)
        print(
            f"RESETTING MASKS module conv1 {pl_module.conv1.weight[0][0]} ")

    def on_after_backward(self, trainer, pl_module):
        for module in pl_module.modules():
            if hasattr(module, "weight_mask"):
                weight = next(param for name, param in module.named_parameters() if "weight" in name)
                weight.grad = weight.grad * module.weight_mask

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f" should print whars here {trainer.logged_metrics}")


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
