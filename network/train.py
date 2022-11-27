import pytorch_lightning as pl
from dataloader import VoxelFutureDataModule
from autoencoder import Autoencoder
import torch
from torchinfo import summary
print(torch.cuda.is_available())
autoencoder = Autoencoder()
data_module = VoxelFutureDataModule(num_workers=8)

trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100)
trainer.fit(model=autoencoder, datamodule=data_module)

#TODO: Test if this works
#TODO: Add proper logging
#TODO: improve hparams