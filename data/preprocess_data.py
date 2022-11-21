import json
import os
import numpy as np
import pickle
import argparse
import sys
from base_dataset import Dataset
from threed_future_labels import THREED_FUTURE_LABELS
from threed_future_model import VoxelThreedFutureModel
from utils import lower_slash_format
import logging

class ThreedFutureDatasetParser(Dataset):
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
        # Set property variables
        self.objects = []
        self.object_type_frequencies = {}
        self.obj_count = 0

        future_labels = THREED_FUTURE_LABELS.values()
        future_keys = THREED_FUTURE_LABELS.keys()
        for v in future_labels:
            self.object_type_frequencies[v] = 0
        for model in self.model_info:
            if model["category"] and lower_slash_format(model["category"]) in future_keys:
                self.objects.append(
                    VoxelThreedFutureModel(
                        model_jid = model["model_id"],
                        model_info= model,
                        scale=1,
                        path_to_models=self.root_path
                    )
                )
                self.object_type_frequencies[THREED_FUTURE_LABELS[lower_slash_format(model["category"])]] +=1
                self.obj_count += 1
        self.object_types = THREED_FUTURE_LABELS
        self.n_object_types = len(THREED_FUTURE_LABELS)

    def __len__(self):
        return self.obj_count

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

def pickle_threed_future_dataset(data):
    objects = []
    print("Parsing dataset ", end="")
    for i in range(len(data)):
    # for i in range(10):
        model = data[i]
        matrix = model.get_voxel_obj_matrix(skip_texture=True)
        objects.append({
            "model_name" : model.model_jid,
            "matrix" : matrix,
            "label" : model.label
        })
        s = "{:5d} / {:5d}".format(i, len(data))
        print(s, flush=True, end="\b"*len(s))
    print()
    pickle.dump(objects, open("/tmp/threed_future.pkl", "wb"))
    print("Data saved at: /tmp/threed_future.pkl")

def serialize_stats(data, output_dir):
    frequencies = data.object_type_frequencies
    stats = {
        "n_objects" : len(data),
        "n_object_types" : len(THREED_FUTURE_LABELS),
        "object_types" : THREED_FUTURE_LABELS,
        "object_type_frequencies" : frequencies
    }
    with open(os.path.join(output_dir,'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)



def main(argv):

    MODEL_SUFFIX =  "model_info.json"
    model_info = None
    
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FUTURE objects to train our model"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/future_voxels",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    args = parser.parse_args(argv)

    root_path = args.path_to_3d_future_dataset_directory
    with open(os.path.join(root_path, MODEL_SUFFIX)) as f:
        model_info = json.load(f)
        
    # We need model data
    assert(model_info != None)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    data =  ThreedFutureDatasetParser(root=root_path)
    
    serialize_stats(data, args.output_directory)
    # Supress trimesh logging
    logger = logging.getLogger("trimesh")
    logger.setLevel(logging.ERROR)
    pickle_threed_future_dataset(data)

    

if __name__ == "__main__":
    main(sys.argv[1:])