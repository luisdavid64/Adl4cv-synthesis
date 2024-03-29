from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import pickle
import argparse
import sys
from base_dataset import Dataset
from threed_future_labels import THREED_FUTURE_LABELS
from threed_future_model import VoxelThreedFutureModel
from utils import lower_slash_format
import logging
from tqdm import tqdm

# Supress trimesh logging in all processes
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


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

def process_object(model):
    matrix = model.get_voxel_matrix(skip_texture=True)
    return ({
        "model_name" : model.model_jid,
        "label" : model.label,
        "matrix" : matrix,
    })


def pickle_threed_future_dataset_parallel(data, output_dir):

    num_cores = (multiprocessing.cpu_count() - 1) or 1 
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(data)) as progress:
            progress.set_description("Processing dataset")
            futures = []
            for model in data:
                future = pool.submit(process_object, model)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            objects = []
            for future in futures:
                object = future.result()
                objects.append(object)
    with open(os.path.join(output_dir,'threed_future.pkl'), 'wb') as f:
        pickle.dump(objects, f)
        print(f"Data saved at: {f.name}")

def pickle_threed_future_dataset(data, output_dir):
    objects = []
    print("Parsing dataset ", end="")
    for i in range(len(data)):
        model = data[i]
        matrix = model.get_voxel_matrix(skip_texture=True)
        objects.append({
            "model_name" : model.model_jid,
            "label" : model.label,
            "matrix" : matrix,
        })
        s = "{:5d} / {:5d}".format(i, len(data))
        print(s, flush=True, end="\b"*len(s))
    print()
    with open(os.path.join(output_dir,'threed_future.pkl'), 'wb') as f:
        pickle.dump(objects, f)
        print(f"Data saved at: {f.name}")

def pickle_dataset(data, output_dir, parallelize=False):
    if parallelize:
        pickle_threed_future_dataset_parallel(data, output_dir)
    else:
        pickle_threed_future_dataset(data, output_dir)


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
        "--output_directory",
        default="../../output/",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        default="../../ATISS/datasets/3D-Front/3D-FUTURE-model",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument('-p', action='store_true')
    args = parser.parse_args(argv)

    root_path = args.path_to_3d_future_dataset_directory
    with open(os.path.join(root_path, MODEL_SUFFIX)) as f:
        model_info = json.load(f)
        
    # We need model data
    assert model_info != None, "3D-Future model information needed."

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    data =  ThreedFutureDatasetParser(root=root_path)

    if args.p:
        print("--- Processing dataset in parallel ---")
    else:
        print("--- Processing dataset sequentially ---")

    # Produce json file for statistics file in output directory
    serialize_stats(data, args.output_directory)
    # Pickle dataset to /tmp/threed_future.pkl
    pickle_dataset(data, args.output_directory ,args.p)

    

if __name__ == "__main__":
    main(sys.argv[1:])