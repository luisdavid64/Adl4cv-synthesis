"""Dataset Base Class"""

from abc import ABC, abstractmethod

from download_utils import download_dataset


class Dataset(ABC):
    def __init__(self, root, download_url=None, force_download=False):
        self.root_path = root
        if download_url is not None:
            dataset_zip_name = download_url[download_url.rfind('/')+1:]
            self.dataset_zip_name = dataset_zip_name
            download_dataset(
                url=download_url,
                data_dir=root,
                dataset_zip_name=dataset_zip_name,
                force_download=force_download,
            )

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""