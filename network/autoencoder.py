from matplotlib import pyplot as plt
import torch
from torch import nn 
import torch.nn.functional as F
import pytorch_lightning as pl

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
    def __init__(self, hparams=None):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.save_hyperparameters(hparams)
        self.step = 0

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
        self.step = self.step + 1

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.binary_cross_entropy(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_val_loss", avg_loss)
        return avg_loss

    def test_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.binary_cross_entropy(x_hat, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, x_hat, "test_input_output")
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_test_loss", avg_loss)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

    def save_images(self, x, output, name, n=4):
        """
        Saves a plot of n images from input and output batch
        """

        if self.hparams.batch_size < n:
            raise IndexError("You are trying to plot more images than your batch contains!")
        fig = plt.figure()
        fig.suptitle('Voxel reconstruction ' + str(self.step))
        # Plot input images
        for i in range(n):
            ax = fig.add_subplot(2, n, i + 1, projection='3d')
            ax.view_init(azim=-60, elev=120)
            ax.voxels(x[i][0])
            
        # Plot reconstruction images
        for i in range(n):
            ax = fig.add_subplot(2, n, n + i + 1, projection='3d')
            ax.view_init(azim=-60, elev=120)
            # Round to nearest integer representation
            ax.voxels(torch.round(output[i][0]))
        self.logger.experiment.add_figure(name , fig)

