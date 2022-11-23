import numpy as np

import sys, os

# Add data folder to system path for tests
testdir = os.path.dirname(__file__)
srcdir = '../data'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import threed_future_dataset

# Check that pickled dataset is 32x32x32
dataset, stats = threed_future_dataset.ThreedFutureDataset.from_pickled_dataset("/tmp/threed_future.pkl", "../data/output/dataset_stats.json")
for i in range(len(dataset)):
    assert np.array_equal(dataset[i]["matrix"].shape, np.array([32,32,32])), "Some data is not 32x32x32."
print("Success: Dataset is in the right format.")