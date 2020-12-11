import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LitManualAutoEncoder(pl.LightningModule):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        opt_a = self.optimizers()

        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        # backward acts like normal backward
        self.manual_backward(loss, opt_a, retain_graph=True)
        self.manual_backward(loss, opt_a)
        opt_a.step()
        opt_a.zero_grad()

        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss,prog_bar=True)
        # --------------------------

    def test_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer