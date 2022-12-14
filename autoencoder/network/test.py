from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from autoencoder import Autoencoder
from dataloader import VoxelFutureDataModule


def main(hparams):
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    autoencoder = Autoencoder(hparams)
    autoencoder.load_state_dict(torch.load("./output/autoencoder.pt"))
    logger = pl.loggers.TensorBoardLogger(hparams.log_dir, name=f"bs{hparams.batch_size}")
    trainer = pl.Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs, log_every_n_steps =  1)
    data_module = VoxelFutureDataModule(data_dir=hparams.data_root, num_workers=hparams.num_workers, overfit=20)
    trainer.test(model=autoencoder, datamodule=data_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--data_root", type=str, default="/tmp/threed_future.pkl", help="Data root directory")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    args = parser.parse_args()
    main(args)