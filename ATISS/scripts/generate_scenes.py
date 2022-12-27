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

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects, get_textured_objects_gt, get_textured_objects_from_voxels
import seaborn as sns
import trimesh

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
        default='../outputs/Y8V2L5S9M/model_05800',
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
    parser.add_argument(
        "--shape_codes_path",
        default="../../output/threed_future_encoded_shapes.pkl",
        help="Path to encodes shapes"
    )
    parser.add_argument(
        "--shape_generator_model_path",
        default="../../autoencoder/network/output/pretrained_ae.pt",
        help="Path to encodes shapes"
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
        shape_codes_path=args.shape_codes_path
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    autoencoder = Autoencoder({"z_dim": 128})
    autoencoder.load_state_dict(torch.load(args.shape_generator_model_path))
    autoencoder.freeze()

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    classes = np.array(dataset.class_labels)
    color_palette = np.array(sns.color_palette('hls', len(classes)-2))
    for i in range(args.n_sequences):
        scene_idx = given_scene_id or np.random.choice(len(dataset))
        current_scene = raw_dataset[scene_idx]
        print("{} / {}: Using the {} floor plan of scene {}".format(
            i, args.n_sequences, scene_idx, current_scene.scene_id)
        )
        room_mask = torch.from_numpy(
            np.transpose(current_scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
        # room_mask = room_mask.cuda()
        room_mask = room_mask
        bbox_params = network.generate_boxes(room_mask=room_mask, device=device)
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"],
        ], dim=-1).cpu().numpy()
        voxel_shapes_t = autoencoder.decoder(torch.squeeze(boxes["shape_codes"],dim=0))

        # This generates our ground truth.
        # Are we going to generate our Autoencoder ground truth or real gt?
        bbox_params_gt = np.concatenate([
            current_scene.class_labels[None],
            current_scene.translations[None],
            current_scene.sizes[None],
            current_scene.angles[None]
        ], axis=-1)
        voxel_shapes_gt = autoencoder.decoder(torch.stack(current_scene.shape_codes))

        # renderables, trimesh_meshes = get_textured_objects_from_voxels(
        #     bbox_params_gt, classes, voxel_shapes_t[None]
        # )

        renderables_gt, trimesh_meshes_gt = get_textured_objects_from_voxels(
            bbox_params_gt, classes, voxel_shapes_gt[None]
        )
        # if trimesh_meshes is not None:
        #     # Create a trimesh scene and export it
        #     path_to_objs = os.path.join(
        #         args.output_directory, 'pred',
        #         "{:03d}_scene".format(i)
        #     )
        #     if not os.path.exists(path_to_objs):
        #         os.makedirs(path_to_objs)
        #     export_scene(path_to_objs, trimesh_meshes, boxes["class_labels"][0], classes, color_palette)

        if trimesh_meshes_gt is not None:
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory, 'gt',
                "{:03d}_scene_gt".format(i)
            )
            if not os.path.exists(path_to_objs):
                os.makedirs(path_to_objs)
            export_scene_gt(path_to_objs, trimesh_meshes_gt, current_scene.class_labels, classes, color_palette)

def export_scene(path_to_objs, trimesh_meshes, class_labels, classes, color_palette):
    for (inst_idx, trimesh_mesh) in zip(range(1, class_labels.shape[0]-1), trimesh_meshes):
        cls_label = class_labels[inst_idx].argmax()
        color = color_palette[cls_label]
        ply_file = path_to_objs + '/%d_%s.ply' % (inst_idx, classes[cls_label])
        trimesh_mesh = trimesh.Trimesh(vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces, vertex_colors=color)
        trimesh_mesh.export(ply_file)

def export_scene_gt(path_to_objs, trimesh_meshes, class_labels, classes, color_palette):
    for (inst_idx, trimesh_mesh) in zip(range(class_labels.shape[0]), trimesh_meshes):
        cls_label = class_labels[inst_idx].argmax()
        color = color_palette[cls_label]
        ply_file = path_to_objs + '/%d_%s.ply' % (inst_idx, classes[cls_label])
        trimesh_mesh = trimesh.Trimesh(vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces, vertex_colors=color)
        trimesh_mesh.export(ply_file)

if __name__ == "__main__":
    main(sys.argv[1:])
