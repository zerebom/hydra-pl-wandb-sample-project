import warnings

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.factory.dataset import DataModule
warnings.filterwarnings("ignore")

@hydra.main(config_path='../config', config_name='default_config')
def train(cfg: DictConfig) -> None:
    model = instantiate(cfg.model.instance,cfg=cfg)

    dm = DataModule(cfg.data)
    dm.setup()

    wandb_logger = instantiate(cfg.logger)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    early_stopping = instantiate(cfg.callbacks.EarlyStopping)
    model_checkpoint = instantiate(cfg.callbacks.ModelCheckpoint)
    wandb_image_logger = instantiate(cfg.callbacks.WandbImageLogger,
                            val_imgs=next(iter(dm.val_dataloader()))[0])

    trainer = pl.Trainer(
        logger = wandb_logger,
        checkpoint_callback = model_checkpoint,
        callbacks=[early_stopping,wandb_image_logger],
        **cfg.trainer.args,
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    train()