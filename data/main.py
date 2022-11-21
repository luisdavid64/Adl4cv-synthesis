from threed_future_dataset import ThreedFutureDataset
import os

root = os.environ.get("FUTURE_DATASET")
if not root:
    root = "/Users/luisreyes/Downloads/3D-FUTURE-model"
# dataset = ThreedFutureDataset(root=root)
# dataset[0].show_voxel_plot()

# Import pickled dataset
dataset, stats = ThreedFutureDataset.from_pickled_dataset("/tmp/threed_future.pkl", "/Users/luisreyes/Courses/ADL4CV/Project/data/output/dataset_stats.json")
print(len(dataset))