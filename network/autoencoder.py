from matplotlib import pyplot as plt
import torch
from torch import nn 
import torch.nn.functional as F
import pytorch_lightning as pl
from mpl_toolkits.mplot3d import Axes3D
import torchvision.utils as vutils

input = (1,32,32,32)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
          nn.Conv3d(1, 30, 5, padding=2),
          nn.ReLU(),
          torch.nn.MaxPool3d(2),
          nn.Conv3d(30, 60, 5, padding=2),
          nn.ReLU(),
          torch.nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
          nn.Conv3d(60, 60, 5, padding=2),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv3d(60, 30, 5, padding=2),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv3d(30, 1, 5, padding=2),
          nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
  

class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.hparams = hparams

    def forward(self, x):
        reconstruction = self.decoder(self.encoder(x))
        return reconstruction

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.binary_cross_entropy(x_hat, x)

        # # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, x_hat, "train_input_output")

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.binary_cross_entropy(x_hat, x)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.binary_cross_entropy(x_hat, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, x_hat, "test_input_output")

        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"avg_test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

    def save_images(self, x, output, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """

        if self.hparams.batch_size < n:
            raise IndexError("You are trying to plot more images than your batch contains!")
        grid_top = vutils.make_grid(x, nrow=n)
        grid_bottom = vutils.make_grid(output, nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid)


