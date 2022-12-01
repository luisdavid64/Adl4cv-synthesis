import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class VoxelFutureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/tmp/threed_future.pkl", batch_size: int = 128, num_workers:int = 8, overfit = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.overfit = overfit

    def filter_dataset(self, data, filter_label="desk"):
            return filter(lambda x: x["label"] == filter_label,data)

    def setup(self, stage: str):
        with open(self.data_dir, "rb") as f:
            dataset = pickle.load(f)
            print(len(dataset))
            print(type(dataset))
            dataset = list(map(lambda x: torch.from_numpy(x["matrix"]).float().unsqueeze(0), self.filter_dataset(dataset)))
            # Overfit to a certain number of samples for testing
            if self.overfit != None:
                dataset = dataset[0:self.overfit]
            train_size = int(0.8*len(dataset))
            val_size = int(len(dataset)-train_size)
            self.train, self.val = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...