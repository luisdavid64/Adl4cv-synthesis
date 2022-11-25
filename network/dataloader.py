import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class VoxelFutureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/tmp/threed_future.pkl", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self):
        with open(self.data_dir, "rb") as f:
            dataset = pickle.load(f)
            self.train, self.val = torch.utils.data.random_split(dataset, [0.8,0.2])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...