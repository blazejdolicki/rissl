# Self-supervised learning with group equivariant networks for histopathological images
Repository containing code for my MSc thesis "Self-supervised learning with group
equivariant networks for histopathological
images". 

You can read my thesis here: [Master Thesis - Blazej Dolicki - Final.pdf](Master%20Thesis%20-%20Blazej%20Dolicki%20-%20Final.pdf)


## Installation
Keep in mind that the commands mentioned below apply for Linux and will often not work on other operating systems.

1. Clone my fork of VISSL: https://github.com/blazejdolicki/vissl. The two most important locations of this fork is [configs/config/pretrain/moco](https://github.com/blazejdolicki/vissl/tree/main/configs/config/pretrain/moco) which contain configs for MoCo models that I used for pretraining and [custom_catalog.json](https://github.com/blazejdolicki/vissl/blob/main/custom_catalog.json).
2. Clone this repository, such that these two repos are on the same level in a directory tree, for instance:
```
thesis/
    rissl/ <- this repo
    vissl/ <- fork of VISSL
```

## How to pretrain with new data?
After following the steps above, you can add pretraining models with new data:
1. Create a script that converts the original data into a VISSL-compatible [data source](https://vissl.readthedocs.io/en/v0.1.5/vissl_modules/data.html#reading-data-from-several-sources). An example of such script is `create_tcga_crc_filelist.py`. Run the script:
```
python data_scripts/create_tcga_crc_filelist.py --data_dir /mnt/archive/data/pathology/TCGA_Kather_preprocessed/images/ffpe
```

2. Register your new dataset in [custom_catalog.json](https://github.com/blazejdolicki/vissl/blob/main/custom_catalog.json). This is required by VISSL. However, you can just add a placeholder like below and later define the paths in your job script.
```
"tcga_crc": {
   "train": ["", ""]
}

3. 
```

## Datasets
Descriptions of the datasets are available in the paper. Most of datasets are compressed in some format and you have to unpack them.
For `.zip` files  (given the current dir contains `<file>`:
```
unzip <file>.zip
```
For `.tar.gz` files (given the current dir contains `<file>`:
```
tar -xvf <file>.tar.gz
```
Once you unpack the data, you can remove the compressed file to save disk space with `rm`.
### PatchCam
Download all files listed [in this repository](https://github.com/basveeling/pcam) and unpack in `data/pcam` (TODO: probably later update directory).
Convert the dataset from `.h5` files to a format compatible with VISSL:
```
 python data_scripts/create_patch_camelyon_data_files.py -i data/pcam -o data/pcam
```
### NCT-CRC-HE-100K
Download all files from [here](https://zenodo.org/record/1214456#.YaCjaNDMJPa) and unpack in `data/nct`.
Create `.npy` files of data splits which are compatible with VISSL:
```
python data_scripts/create_nct_filelist.py --data_dir data/nct
```
### BreaKHis
Download [here](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and unpack in `data/breakhis`.

In order to use cross validation with 5 folds from the original paper, download the python script and once the data is downloaded and unpacked, run:
```
python mkfold.py
```

Afterwards, create a directory `data/breakhis_fold` and move folders `fold1`, `fold2` etc. there.

However, each of the above folds contains only training and test set, without a validation set which is required for selecting the best epoch or hyperparameter tuning. Therefore, we use a different split. In our experiments, the whole dataset is split into a training, validation and test set consisting of 70%, 20% and 10% of the dataset respectively. In order to create this split run:
```
python data_scripts/create_breakhis_filelist.py --data_dir data/breakhis
```

### Bach
Download [here](https://zenodo.org/record/3632035#.YbdBDr3MJPa) and unpack in `data/bach`.

## Mean Rotation Error
To test our implementation of the Mean Rotation Error, activate the conda environment, go to the `rissl` directory and run:
```python
$ python test_mre.py
```
the expected output is:
```bash
2022-06-16 13:28:33,911 [INFO] Random seed: 7
2022/06/16 13:28:33 INFO mlflow.tracking.fluent: Experiment with name 'test_mre' does not exist. Creating a new experiment.
2022-06-16 13:28:34,460 [INFO] Train size 10
2022-06-16 13:28:37,435 [INFO] Model resnet18 initialized from scratch.
2022-06-16 13:28:38,227 [INFO] MRE(4):  0.0
Test passed
```
