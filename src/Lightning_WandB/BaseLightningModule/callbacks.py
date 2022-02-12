from pytorch_lightning.callbacks import Callback
import copy

try:
    from ..utils import apply_pruning, count_rem_weights
except ImportError:
    from src.Lightning_WandB.utils import apply_pruning, count_rem_weights


# Full model to be trained. And final and Initial weights stored
class FullTrainer(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # If module has reset epoch is same as current epoch, update the weights dict
        if trainer.current_epoch == pl_module.hparams.reset_epoch:
            pl_module.original_wgts = copy.deepcopy(pl_module.state_dict())


# prune model before training.
class PruneTrainer(Callback):
    def on_train_start(self, trainer, pl_module):
        # before the new training run happens, prune the model and reset the weights
        apply_pruning(pl_module, 0.1)
        pl_module.reset_weights()
        pl_module.test_model_change()
