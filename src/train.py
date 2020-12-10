import warnings

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.factory.dataset import DataModule
warnings.filterwarnings("ignore")

@hydra.main(config_path='../config', config_name='sample')
def train(cfg: DictConfig) -> None:
    model = instantiate(cfg.model.instance)

    dm = DataModule(cfg.data)
    dm.setup()

    wandb_logger = instantiate(cfg.logger)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    early_stopping = instantiate(cfg.callbacks.EarlyStopping)
    model_checkpoint = instantiate(cfg.callbacks.ModelCheckpoint)

    trainer = pl.Trainer(
        logger = wandb_logger,
        checkpoint_callback = model_checkpoint,
        #wandb_callback
        callbacks=[early_stopping],
        **cfg.trainer.args,
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    train()