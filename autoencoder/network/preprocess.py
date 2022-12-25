import itertools
from argparse import ArgumentParser
import pickle
from autoencoder import Autoencoder
import torch
import trimesh
from matplotlib import pyplot as plt
import random
from tqdm import tqdm

def main(hparams):
    autoencoder = Autoencoder(hparams)
    autoencoder.load_state_dict(torch.load(hparams["pretrained_model_path"]))
    autoencoder.eval()
    with open(hparams["data_root"], "rb") as f:
        dataset = pickle.load(f)
        dataset = list(map(lambda x: {'model_name' : x["model_name"], 'matrix' :torch.from_numpy(x["matrix"]).float().unsqueeze(0)}, dataset))
        print(len(dataset))
        k = random.randint(0, len(dataset) - 1)

        shapes = {}

        for object in tqdm(dataset):
            x = autoencoder.encoder(torch.unsqueeze(object['matrix'],0))
            shapes[object['model_name']] = x
        pickle.dump(shapes, open("../../output/threed_future_encoded_shapes.pkl", "wb"))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../tmp/threed_future.pkl", help="Data root directory")
    parser.add_argument("--z_dim", type=int, default=128, help="Size of latent vector z")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--pretrained_model_path", type=str, default="output/pretrained_ae.pt", help="Pretrained model location")
    args = vars(parser.parse_args())
    main(args)