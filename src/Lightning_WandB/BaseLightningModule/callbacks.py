from pytorch_lightning.callbacks import Callback


# Full model to be trained. And final and Initial weights stored
class FullTrainer(Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.save_checkpoint(f"{pl_module.hparams.experiment_dir}/init_trainer_weights.ckpt")
