from matplotlib import pyplot as plt
import torch
from torch import nn 
import torch.nn.functional as F
import pytorch_lightning as pl

input = (1,32,32,32)

# Custom reshape layer
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

# GAE loss function
# class GeneralizedLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, batch_size=16):
#         super(nn.MSELoss, self).__init__()
#         self.batch_size = batch_size

#     def chamfer_distance(self, inputs : torch.Tensor, targets : torch.Tensor):
#         inputs = inputs.to_sparse()
#         targets = targets.to_sparse()
#         return chamfer_dist(inputs,targets)

#     def forward(self, inputs, targets, smooth=1):
#         loss = 0
#         loss += F.mse_loss(inputs, targets)
#         dist1, dist2, idx1, idx2 = chamfer_dist(inputs,targets)
#         for i in range(self.batch_size):
#             pass
#             # curr_idx = indices[i]
#             # sort_idx = tf.gather(sort_distance_idx, curr_idx)
#             # data_true = tf.gather(data_train, sort_idx[0, 0])
#             # for j in range(10):
#             #     curr_data_train = tf.gather(data_train, sort_idx[0, j])
#             #     s = tf.math.exp(-(tf.norm(data_true - curr_data_train) ** 2) / 200)
#             #     loss += s * mean_squared_error(curr_data_train, y_pred[i, :]) 
#         return loss

class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = nn.Sequential(
          nn.Conv3d(1, 32, 4, stride=2, padding=1),
          nn.BatchNorm3d(32),
          nn.ReLU(),
          nn.Conv3d(32, 16, 4, stride=2, padding=1),
          nn.BatchNorm3d(16),
          nn.ReLU(),
          nn.Conv3d(16, 8, 4, stride=2, padding=1),
          nn.BatchNorm3d(8),
          nn.ReLU(),
          nn.Flatten(start_dim=1),
          nn.Linear(in_features=512, out_features=hparams["z_dim"]),
          nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Input needs to be reshaped
        self.decoder = nn.Sequential(
          nn.Linear(in_features=hparams["z_dim"], out_features=512),
          View((-1,8,4,4,4)),
          nn.ConvTranspose3d(8, 16, 4, stride=2, padding=1),
          nn.BatchNorm3d(16),
          nn.ReLU(),
          nn.ConvTranspose3d(16, 32, 4, stride=2, padding=1),
          nn.BatchNorm3d(32),
          nn.ReLU(),
          nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),
          nn.BatchNorm3d(1),
          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)
  

class Autoencoder(pl.LightningModule):
    def __init__(self, hparams=None, log_images=False):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.save_hyperparameters(hparams)
        self.step = 0
        self.log_images = log_images

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)

        # # save input and output images at beginning of epoch
        if self.log_images and batch_idx == 0:
            self.save_images(x, x_hat, "train_input_output")
        self.step = self.step + 1

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_val_loss", avg_loss)
        return avg_loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], betas=(self.hparams["beta1"], self.hparams["beta2"]))

    def save_images(self, x, output, name, n=4):
        """
        Saves a plot of n images from input and output batch
        """

        if self.hparams["batch_size"] < n:
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

