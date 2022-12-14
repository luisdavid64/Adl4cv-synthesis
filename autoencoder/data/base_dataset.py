"""Dataset Base Class"""

from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, root, download_url=None, force_download=False):
        self.root_path = root

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""