import json
import os
import numpy as np
import pickle

import torch

from base_dataset import Dataset
from threed_future_model import ThreedFutureModel

class ThreedFutureDataset(Dataset):
    def __init__(self, *args,
                root=None,
                labels=None,
                transform=None,
                download_url="",
                **kwargs):
        super().__init__(*args,
                            download_url=download_url,
                            root=root,
                            **kwargs)
        
        #File mapping categorical info to objects
        self.model_suffix =  "model_info.json"
        with open(os.path.join(self.root_path,self.model_suffix)) as f:
            self.model_info = json.load(f)
        #Transformations on data
        self.transform = transform
        self.objects = []
        for model in self.model_info:
            self.objects.append(
                ThreedFutureModel(
                    model_jid = model["model_id"],
                    model_info= model,
                    scale=1,
                    path_to_models=self.root_path
                )
            )



    def __len__(self):
        return len(self.model_info)

    def __getitem__(self, index):
        object = self.objects[index]
        if self.transform is not None:
            object = self.transform(object)
        return object
    
    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]