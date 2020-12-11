from pytorch_lightning.callbacks import Callback
import wandb

class ImagePredictionLogger(Callback):
    def __init__(self,val_imgs,num_samples=32):
        super().__init__()
        self.val_imgs = val_imgs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)

        # Get model prediction
        pred_imgs = pl_module(val_imgs).cpu().numpy().reshape(-1,28,28)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x) for x in pred_imgs[:self.num_samples]]
        })