import itertools
from argparse import ArgumentParser
import pickle
from autoencoder import Autoencoder
import torch
import trimesh
from matplotlib import pyplot as plt
import random
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
import numpy as np
import math

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

def main(hparams):
    with torch.no_grad():
        autoencoder = Autoencoder(hparams)
        autoencoder.load_state_dict(torch.load(hparams["pretrained_model_path"]))
        autoencoder.eval()
        with open(hparams["data_root"], "rb") as f:
            dataset = pickle.load(f)
            labels = []
            sums = []
            counts = []
            dataset = list(map(lambda x: {'model_name' : x["model_name"], 'matrix' :torch.from_numpy(x["matrix"]).float().unsqueeze(0), 'label' : x["label"]}, dataset))
            # print(len(dataset))

            for object in tqdm(dataset):
                if object['model_name'] == "ec00b288-3958-4434-94fa-5e6663c8be9a": # model is effectively empty
                    continue
                x = voxels_to_points(torch.round(torch.squeeze(autoencoder(torch.unsqueeze(object['matrix'],0)))))
                #getdist
                # dist = 2.
                y = voxels_to_points(torch.squeeze(object['matrix']))
                # print(x.shape)
                # print(y.shape)
                # break
                dist = eval_object(x,y)
                if math.isnan(dist):
                    print(dist)
                    print(x)
                    print(y)
                    print(object)
                    print(torch.squeeze(object['matrix']))
                try:
                    i = labels.index(object['label'])
                    sums[i] += dist
                    counts[i] += 1.
                except ValueError:
                    labels.append(object['label'])
                    sums.append(dist)
                    counts.append(1.)
            sums_np = np.array(sums)
            counts_np = np.array(counts)
            mean = sums_np/counts_np
            print(list(zip(labels,counts,mean)))
            # pickle.dump(shapes, open("../../output/threed_future_encoded_shapes.pkl", "wb"))

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