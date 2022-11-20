from threed_future_dataset import ThreedFutureDataset
import os

root = os.environ.get("FUTURE_DATASET")
if not root:
    root = "/Users/luisreyes/Downloads/3D-FUTURE-model"
dataset = ThreedFutureDataset(root=root)
dataset[11].show_voxel_plot(use_texture=True)