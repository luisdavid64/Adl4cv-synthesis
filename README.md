## End2End Scene Synthesis

This is the project repository for the Advanced Deep Learning for Computer Vision project.


## Installation & Dependencies

The simplest way to make sure that you have all dependencies in place is to use
[conda](https://docs.conda.io/projects/conda/en/4.6.1/index.html). You can
create a conda environment called ```project``` using
```
conda env create -f environment.yaml
conda activate project
```
## Dataset
The project is based on the 
[3D-FUTURE](https://www.google.com/search?q=3d-future&oq=3d-fut&aqs=chrome.1.69i57j0j0i30l8.3909j0j7&sourceid=chrome&ie=UTF-8)
dataset. The dataset can be obtained in [this webpage](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).

### Data Preprocessing

Once you have downloaded the dataset you need to run the `preprocess_data.py` script in order to prepare the data to be able to train a model. To run the preprocessing script run

```
python preprocess_data.py path_to_output_dir path_to_3d_future_dataset_dir -p
```

The optional p flag tries to parallelize the preprocessing step for better performance. The preprocessing script takes the 3D-FUTURE dataset and transforms the data to a list of objects consisting of:
- model name
- label
- $32\times 32 \times 32$ voxel grid