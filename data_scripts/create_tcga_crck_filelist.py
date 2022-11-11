import os
import random
import numpy as np
import argparse

def get_filepaths_recursively(root_path, shuffle=True):
    file_paths = []
    # only loop through folders with training data for colorectal cancer
    folders = ["CRC_DX_TRAIN_MSIMUT/MSIMUT", "CRC_DX_TRAIN_MSS/MSS"]
    for folder in folders:  # for each folder
        folder_path = os.path.join(root_path, folder)
        folder_imgs = os.listdir(folder_path)
        folder_imgs = [os.path.join(folder_path, img) for img in folder_imgs]
        file_paths += folder_imgs

    file_paths = np.array(file_paths)

    # shuffle to mix the classes for better training
    if shuffle:
        random.shuffle(file_paths)

    return file_paths


def save_split(imgs, split, data_path):
    filelist_imgs_path = os.path.join(data_path, f"{split}_images.npy")
    print(f"Saving {split} set with {len(imgs)} examples in {filelist_imgs_path}")
    np.save(filelist_imgs_path, np.array(imgs))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sample", action="store_true", help="Use sample of the dataset.")
    parser.add_argument("--sample_size", default=10,
                        type=int, help="Size of the sampled datasets")
    parser.add_argument("--seed", default=7,
                        type=int, help="Random seed used to sample splits to enable reproducibility")
    parser.add_argument("--data_dir", type=str, help="Directory where the data is stored and where"
                                                     "the output files will be store")
    args = parser.parse_args()

    # /mnt/archive/data/pathology/TCGA_Kather_preprocessed/images/ffpe
    DATA_PATH = args.data_dir

    print("Arguments")
    print(args)

    random.seed(args.seed)

    split = "train"
    imgs = get_filepaths_recursively(args.data_dir)

    expected_num_imgs = 93408
    assert expected_num_imgs == len(imgs)

    save_split(imgs, split, args.data_dir)



