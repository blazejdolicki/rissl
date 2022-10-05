from pathlib import Path
import os
import random
import numpy as np
import argparse

def find_imgs_recursively(root_path, img_format=".png"):
    return list(Path(root_path).rglob(f"*{img_format}"))

def get_filepaths_recursively(root_path):
    file_paths, file_labels = [], []
    for cls in os.listdir(root_path):  # for each class
        class_path = os.path.join(root_path, cls)
        if os.path.isdir(class_path): # only loop through folders
            class_imgs = find_imgs_recursively(class_path)
            class_imgs = [os.path.join(cls, str(img)) for img in class_imgs]
            class_labels = [cls] * len(class_imgs)
            file_paths += class_imgs
            file_labels += class_labels

    file_paths, file_labels = np.array(file_paths), np.array(file_labels)

    return file_paths, file_labels

def save_split(imgs, labels, split):
    print(f"Saving {split} set with {len(imgs)} examples")
    # save the filelists
    filelist_imgs_path = os.path.join(OUTPUT_PATH, f"{split}_images.npy")
    filelist_labels_path = os.path.join(OUTPUT_PATH, f"{split}_labels.npy")
    np.save(filelist_imgs_path, np.array(imgs))
    np.save(filelist_labels_path, np.array(labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=7,
                        type=int, help="Random seed used to sample splits to enable reproducibility")
    parser.add_argument("--data_dir", type=str, help="Directory where the data is stored and where"
                                                     "the output files will be store")
    args = parser.parse_args()

    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.2
    TEST_RATIO = 0.1

    DATA_PATH = "BreaKHis_v1/histology_slides/breast"
    DATA_PATH = os.path.join(args.data_dir, DATA_PATH)
    OUTPUT_PATH = args.data_dir
    print("DATA PATH", DATA_PATH)

    # set seed
    random.seed(args.seed)

    imgs, labels = get_filepaths_recursively(DATA_PATH)
    assert len(imgs) == len(labels), "Number of images and labels should be equal"

    # shuffle all image indices
    random_idxs = np.random.permutation(len(imgs))
    # split shuffled indices into 3 lists based on the train-val-test ratios
    split_points = [int(TRAIN_RATIO * len(imgs)), int((TRAIN_RATIO+VALID_RATIO)*len(imgs))]
    train_val_test_idxs = np.split(random_idxs, split_points)
    split_names = ["train", "val", "test"]
    # save filelist for each split
    for split_name, split_idxs in zip(split_names, train_val_test_idxs):
        split_imgs = imgs[split_idxs]
        split_labels = labels[split_idxs]
        assert set(split_labels) == {'benign', 'malignant'}, "Not all image classes present"
        save_split(split_imgs, split_labels, split_name)



