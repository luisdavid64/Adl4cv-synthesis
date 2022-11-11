from threed_future_dataset import ThreedFutureDataset
import matplotlib.pyplot as plt

dataset = ThreedFutureDataset(root="/Users/luisreyes/Downloads/3D-FUTURE-model")

angel_voxel = dataset[10].show()