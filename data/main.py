from threed_future_dataset import ThreedFutureDataset
import matplotlib.pyplot as plt
import os
from threed_future_labels import THREED_FUTURE_LABELS
import numpy as np

root = os.environ.get("FUTURE_DATASET")
if not root:
    root = "/Users/luisreyes/Downloads/3D-FUTURE-model"
dataset = ThreedFutureDataset(root=root)
dataset[8].show_voxel_plot(use_texture=False)