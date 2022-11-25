import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
input = (1,32,32,32)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
          nn.Conv3d(1, 30, 5, padding="same"),
          nn.ReLU(),
          torch.nn.MaxPool3d(2),
          nn.Conv3d(30, 60, 5, padding="same"),
          nn.ReLU(),
          torch.nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
          nn.Conv3d(60, 60, 5, padding="same"),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv3d(60, 30, 5, padding="same"),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv3d(30, 1, 5, padding="same"),
          nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)
  

class Autoencoder(pl.LightningModule):
    def __init__(self, hparams, train_set, val_set):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        reconstruction = self.decoder(self.encoder(x))
        return reconstruction

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.binary_cross_entropy(x, x_hat)
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

