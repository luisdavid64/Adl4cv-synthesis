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
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
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
        "--n_sequences",
        default=1000,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
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

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

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
    # for i in range(args.n_sequences):
    total_distance = 0
    object_count = 0
    with open(config["data"]["shape_codes_path"], "rb") as f:
        shape_codes_dict = pickle.load(f)

    shape_codes_tensor = torch.stack(list(shape_codes_dict.values())).squeeze()
    for i in tqdm(range(len(dataset))):
        scene_idx = i
        current_scene = raw_dataset[scene_idx]
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )
        room_mask=room_mask.to(device)
        bbox_params = network.generate_boxes(room_mask=room_mask, device=device)
        boxes = dataset.post_process(bbox_params)
        closest_shape_codes_id = torch.argmin(torch.cdist(torch.squeeze(boxes["shape_codes"]), shape_codes_tensor), dim=1)
        closest_shape_codes = shape_codes_tensor[closest_shape_codes_id]
        scene_distance = 0
        for i in range(1, closest_shape_codes.shape[0]-1):#skip first and last (start and end symbol)
            voxel_shapes_t = voxels_to_points(torch.round(autoencoder.decoder(torch.squeeze(boxes["shape_codes"])[i]).squeeze()))
            voxel_shapes_gt = voxels_to_points(torch.round(autoencoder.decoder(closest_shape_codes[i]).squeeze()))
            distance = eval_object(voxel_shapes_t, voxel_shapes_gt)
            if distance == np.nan_to_num(distance):
                object_count+=1
                scene_distance+=distance
        total_distance+=scene_distance
            # print(distance)
        # sys.exit()
    print("mean_distance:", total_distance/object_count)
    print("total_distance:", total_distance)




def export_floor_plan(path, floor_plan):
    ply_file = path + '/floor.ply'
    trimesh_mesh = trimesh.Trimesh(vertices=floor_plan.vertices, faces=floor_plan.faces)
    trimesh_mesh.export(ply_file)


def export_scene(path_to_objs, trimesh_meshes, class_labels, classes, color_palette, floor_plan):
    for (inst_idx, trimesh_mesh) in zip(range(1, class_labels.shape[0]-1), trimesh_meshes):
        cls_label = class_labels[inst_idx].argmax()
        color = color_palette[cls_label]
        ply_file = path_to_objs + '/%d_%s.ply' % (inst_idx, classes[cls_label])
        trimesh_mesh = trimesh.Trimesh(vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces, vertex_colors=color)
        trimesh_mesh.export(ply_file)
    export_floor_plan(path_to_objs, floor_plan)

def export_scene_gt(path_to_objs, trimesh_meshes, class_labels, classes, color_palette, floor_plan):
    for (inst_idx, trimesh_mesh) in zip(range(class_labels.shape[0]), trimesh_meshes):
        cls_label = class_labels[inst_idx].argmax()
        color = color_palette[cls_label]
        ply_file = path_to_objs + '/%d_%s.ply' % (inst_idx, classes[cls_label])
        trimesh_mesh = trimesh.Trimesh(vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces, vertex_colors=color)
        trimesh_mesh.export(ply_file)
    export_floor_plan(path_to_objs, floor_plan)

if __name__ == "__main__":
    main(sys.argv[1:])
