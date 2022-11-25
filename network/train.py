import pytorch_lightning as pl
from dataloader import VoxelFutureDataModule
from network import Autoencoder

autoencoder = Autoencoder()

data_module = VoxelFutureDataModule()
train_loader = data_module.train_dataloader()

trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

#TODO: Test if this works
#TODO: Add proper logging
#TODO: improve hparams