from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pl_bolts.datasets import DummyDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_shape = cfg.data_shape
        self.num_samples = cfg.num_samples
        self.batch_size = cfg.batch_size

    def setup(self, stage=None):
        self.train_ds = DummyDataset(self.data_shape,(self.num_samples,))
        self.val_ds = DummyDataset(self.data_shape,(self.num_samples,))
        self.test_ds = DummyDataset(self.data_shape,(self.num_samples,))

    def train_dataloader(self):
        return DataLoader(self.train_ds,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=self.batch_size)