from threed_future_dataset import ThreedFutureDataset
import matplotlib.pyplot as plt
import os

root = os.environ.get("FUTURE_DATASET")
dataset = ThreedFutureDataset(root=root)

angel_voxel = dataset[10].show()