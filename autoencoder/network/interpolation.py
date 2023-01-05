from argparse import ArgumentParser
import os
from autoencoder import Autoencoder
import torch
import trimesh
import pickle
from util import filter_data
import trimesh
import random

def marching_cubes(voxel_matrix):
    voxel_matrix = torch.round(voxel_matrix)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_matrix, pitch=1/32)
    return mesh

def save_interpolations(shapes, n=6):
    SAVE_DIRECTORY = "../../output/interpolations/"
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
    for i in range(n):
        mesh = marching_cubes(shapes[i])
        mesh.export(SAVE_DIRECTORY + "interpolation" + str(i) + ".stl")

def main(hparams):
    autoencoder = Autoencoder(hparams)
    autoencoder.load_state_dict(torch.load(hparams["pretrained_model_path"]))
    autoencoder.freeze()
    with open(hparams["data_root"], "rb") as f:
        dataset = pickle.load(f)
        dataset = list(map(lambda x: torch.from_numpy(x["matrix"]).float().unsqueeze(0), filter_data(dataset, filter_label=hparams["filter_term"])))
        assert len(dataset) != 0, "Length of dataset is empty"
        length_data = len(dataset) - 1
        sample_one = autoencoder.encoder(torch.unsqueeze(dataset[random.randint(0, length_data)], dim = 0))
        sample_two = autoencoder.encoder(torch.unsqueeze(dataset[random.randint(0, length_data)], dim = 0))
        RANGE = hparams["num_objects"]
        decoded_list = []
        for i in range(0,RANGE + 1):
            coef = i/RANGE
            decoded = autoencoder.decoder((1-coef)*sample_one + coef*sample_two)
            decoded_list.append(torch.squeeze(decoded))
        save_interpolations(decoded_list, RANGE + 1)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../tmp/threed_future.pkl", help="Data root directory")
    parser.add_argument("--z_dim", type=int, default=128, help="Size of latent vector z")
    parser.add_argument("--pretrained_model_path", type=str, default="output/pretrained_ae.pt", help="Pretrained model location")
    parser.add_argument("--num_objects", type=str, default=40, help="Number of objects to interpolate")
    parser.add_argument("--filter_term", type=str, default="double_bed", help="Filter items")
    args = vars(parser.parse_args())
    main(args)
