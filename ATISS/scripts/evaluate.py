# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating scenes using a previously trained model."""
import sys

sys.path.append('..')
sys.path.append('../..')
from autoencoder.network.autoencoder import Autoencoder
import argparse
import logging
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
from training_utils import load_config
from utils import floor_plan_from_scene

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects, get_textured_objects_gt, get_textured_objects_from_voxels, get_textured_objects_from_voxels_gt
import seaborn as sns
import trimesh
from scipy.spatial import cKDTree as KDTree

from cfg import shape_codes_dim

def voxels_to_points(voxels):
    indices = torch.nonzero(voxels)
    return indices
def eval_object(points_a, points_b):
    try:
        gen_points_kd_tree = KDTree(points_a)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(points_b)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
    except:
        print("a failed", points_a)
    try:
        # other direction
        gt_points_kd_tree = KDTree(points_b)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(points_a)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
    except:
        print("a failed", points_a)
    
    return gt_to_gen_chamfer + gen_to_gt_chamfer

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        type=str,
        default='../datasets/3D-Front/3D-FRONT-texture',
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default='../outputs/bedroom_200_epochs/model_00200',
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--scene_repetitions",
        default=1,
        type=int,
        help="The number of times each floor plan should be used during evaluation"
    )
    parser.add_argument(
        "--repetitions",
        default=10,
        type=int,
        help="The number of times the evaluation should be run"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"]),
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    autoencoder = Autoencoder({"z_dim": shape_codes_dim})
    autoencoder.load_state_dict(torch.load(config["generator"]["shape_generator_model_path"]))
    autoencoder.freeze()

    classes = np.array(dataset.class_labels)
    with open(config["data"]["shape_codes_path"], "rb") as f:
        shape_codes_dict = pickle.load(f)

    shape_codes_tensor = torch.stack(list(shape_codes_dict.values())).squeeze()
    class_avg_total = None
    reps = args.repetitions
    scene_reps = args.scene_repetitions
    for rep in tqdm(range(reps)):
        class_counts = torch.zeros((len(classes))).to(device)
        class_sums = torch.zeros((len(classes))).to(device)
        for i in tqdm(range(len(dataset))):
            for srep in range(scene_reps):
                scene_idx = i
                current_scene = raw_dataset[scene_idx]
                floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                    current_scene, args.path_to_floor_plan_textures
                )
                room_mask=room_mask.to(device)
                bbox_params = network.generate_boxes(room_mask=room_mask, device=device)
                boxes = dataset.post_process(bbox_params)
                class_ids = torch.nonzero(boxes["class_labels"].squeeze())[:,1].to(device)
                closest_shape_codes_id = torch.argmin(torch.cdist(torch.squeeze(boxes["shape_codes"]), shape_codes_tensor), dim=1)
                closest_shape_codes = shape_codes_tensor[closest_shape_codes_id]
                for i in range(1, closest_shape_codes.shape[0]-1):#skip first and last (start and end symbol)
                    voxel_shapes_t = voxels_to_points(torch.round(autoencoder.decoder(torch.squeeze(boxes["shape_codes"])[i]).squeeze()))
                    voxel_shapes_gt = voxels_to_points(torch.round(autoencoder.decoder(closest_shape_codes[i]).squeeze()))
                    distance = eval_object(voxel_shapes_t, voxel_shapes_gt)
                    if distance == np.nan_to_num(distance):
                        class_counts[class_ids[i]]+=1
                        class_sums[class_ids[i]]+=distance
                    # else: print("xxx")
        class_counts[class_counts==0] = 1
        class_avg = class_sums/class_counts
        if class_avg_total == None:
            class_avg_total = class_avg
        else: class_avg_total += class_avg
        str_out = ", ".join("(%s, %s)" % tup for tup in list(zip(classes, class_avg.tolist())))
        # print()
        with open(os.path.join(args.output_directory, "evaluation.txt"), "a+") as f:
            f.write("Iteration: " + str(rep) + "\n")
            f.write(str_out)
            f.write("\n\n")
    class_avg_total /= reps
    str_out = ", ".join("(%s, %s)" % tup for tup in list(zip(classes, class_avg_total.tolist())))
    with open(os.path.join(args.output_directory, "evaluation.txt"), "a+") as f:
        f.write("Final Result:\n")
        f.write(str_out)
    # print("final:", "".join(list(zip(classes, class_avg_total.tolist()))))
    return class_avg_total

if __name__ == "__main__":
    with torch.no_grad():
        main(sys.argv[1:])
