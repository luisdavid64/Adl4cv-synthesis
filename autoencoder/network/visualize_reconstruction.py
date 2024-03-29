from argparse import ArgumentParser
import pickle
from autoencoder import Autoencoder
import torch
import trimesh
import random
# Marching cubes reconstruction of matrix for sanity check
def marching_cubes(voxel_matrix):
    voxel_matrix = torch.round(voxel_matrix)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_matrix, pitch=1/32)
    mesh.split(only_watertight=True)
    mesh.show()

def main(hparams):
    autoencoder = Autoencoder(hparams)
    autoencoder.load_state_dict(torch.load(hparams["pretrained_model_path"]))
    autoencoder.freeze()
    with open(hparams["data_root"], "rb") as f:
        dataset = pickle.load(f)
        dataset = list(map(lambda x: torch.from_numpy(x["matrix"]).float().unsqueeze(0), dataset))
        k = random.randint(0, len(dataset) - 1)
        my_sample = dataset[k] #next(itertools.islice(dataset, k, k+1))
        x = autoencoder(torch.unsqueeze(my_sample,0))
        # Plot first example
        marching_cubes(x[0][0])

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../tmp/threed_future.pkl", help="Data root directory")
    parser.add_argument("--z_dim", type=int, default=128, help="Size of latent vector z")
    parser.add_argument("--pretrained_model_path", type=str, default="output/pretrained_ae.pt", help="Pretrained model location")
    args = vars(parser.parse_args())
    print(args)
    main(args)