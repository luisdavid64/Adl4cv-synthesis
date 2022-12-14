import json
import os
import numpy as np
import pickle

import torch

from base_dataset import Dataset
from threed_future_model import VoxelThreedFutureModel
from threed_future_labels import THREED_FUTURE_LABELS
from utils import lower_slash_format

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
        self.object_type_frequency = {}
        future_labels = THREED_FUTURE_LABELS.values()
        for v in future_labels:
            self.object_type_frequency[v] = 0
        for model in self.model_info:
            if model["category"] and lower_slash_format(model["category"]) in future_labels:
                self.object_type_frequency[THREED_FUTURE_LABELS[lower_slash_format(model["category"])]] +=1
            self.objects.append(
                VoxelThreedFutureModel(
                    model_jid = model["model_id"],
                    model_info= model,
                    scale=1,
                    path_to_models=self.root_path
                )
            )
        self.object_types = THREED_FUTURE_LABELS
        self.n_object_types = len(THREED_FUTURE_LABELS)

    def __len__(self):
        return len(self.model_info)

    def __getitem__(self, index):
        object = self.objects[index]
        if self.transform is not None:
            object = self.transform(object)
        return object
    
    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]


    @property
    def categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._categories])

    @property
    def super_categories(self):
        return set([
            s.lower().replace(" / ", "/")
            for s in self._super_categories
        ])

    # Use this method to unpickle voxelized data
    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset, stats_path = None):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        if stats_path is not None:
            with open(stats_path) as f:
                stats = json.load(f)
        return dataset, stats