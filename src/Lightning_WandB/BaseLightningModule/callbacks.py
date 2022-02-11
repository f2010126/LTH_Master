from pytorch_lightning.callbacks import Callback
from src.Lightning_WandB.utils import apply_pruning, count_rem_weights


# Full model to be trained. And final and Initial weights stored
class FullTrainer(Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.save_checkpoint(f"{pl_module.hparams.experiment_dir}/init_trainer_weights.ckpt")


# prune model before training.
class PruneTrainer(Callback):

    def on_fit_start(self, trainer, pl_module):
        # before the new training run happens, prune the model and reset the weights
        print(f"In Prune call back Fit start")
        print(f"")
        apply_pruning(pl_module, 0.1)
        pl_module.reset_weights()
        print(
            f"XXX Weights {count_rem_weights(pl_module)} \nPost Train from above \n {pl_module.final_wgts['model.conv1.weight_orig'][0][0]} \n OG \n {pl_module.original_wgts['model.conv1.weight_orig'][0][0]} \n model \n {pl_module.model.conv1.weight[0][0]}")

        pl_module.test_model_change()
