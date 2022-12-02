import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from dataloader import VoxelFutureDataModule
from autoencoder import Autoencoder
import torch
from torchinfo import summary


def main(hparams):
    print("Cuda available: ", torch.cuda.is_available())
    logger = pl.loggers.TensorBoardLogger(hparams.log_dir, name=f"bs{hparams.batch_size}")

    autoencoder = Autoencoder(hparams)
    data_module = VoxelFutureDataModule(data_dir=hparams.data_root, num_workers=hparams.num_workers, overfit=20)

    # print detailed summary with estimated network size
    summary(autoencoder, (hparams.batch_size, 1, hparams.in_size, hparams.in_size, hparams.in_size), device="cpu")

    trainer = pl.Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model=autoencoder, datamodule=data_module)
    if not os.path.exists("output"):
        os.makedirs("output")
    torch.save(autoencoder.state_dict(),"./output/autoencoder.pt")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/tmp/threed_future.pkl", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--in_size", type=int, default=32, help="Size of voxels")
    parser.add_argument("--nz", type=int, default=256, help="Size of latent vector z")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)