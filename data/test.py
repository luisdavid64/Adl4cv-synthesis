from threed_future_model import VoxelThreedFutureModel
from threed_future_dataset import ThreedFutureDataset
import os

root = os.environ.get("FUTURE_DATASET")
if not root:
    root = "/Users/luisreyes/Downloads/3D-FUTURE-model"
dataset = ThreedFutureDataset(root=root)
for i in range(0,20):
    dataset[i].show_voxel_plot()

# Import pickled dataset
# dataset, stats = ThreedFutureDataset.from_pickled_dataset("/tmp/threed_future.pkl", "/Users/luisreyes/Courses/ADL4CV/Project/data/output/dataset_stats.json")
# model = VoxelThreedFutureModel(voxel_object = dataset[11])
# model.marching_cubes()