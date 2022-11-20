import json
import os
import numpy as np
import pickle
import torch
import argparse
import sys
from data.threed_future_model import VoxelThreedFutureModel
from threed_future_labels import THREED_FUTURE_LABELS 
from utils import lower_slash_format

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
    args = parser.parse_args()

    root_path = args["path_to_3d_future_dataset_directory"]
    with open(os.path.join(root_path, MODEL_SUFFIX)) as f:
        model_info = json.load(f)
        
    # We need model data
    assert(model_info != None)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args["output_directory"]):
        os.makedirs(args["output_directory"])

    future_labels = THREED_FUTURE_LABELS.values()

    for model in model_info:
        if model["category"] and lower_slash_format(model["category"]) in future_labels:
            object = VoxelThreedFutureModel(
                model_jid = model["model_id"],
                model_info= model,
                scale=1,
                path_to_models=root_path
            )

            voxel = object.get_voxel_obj_arr()



if __name__ == "__main__":
    main(sys.argv[1:])