import os
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_sample", action="store_true", help="Use sample of the dataset.")
parser.add_argument("--sample_size", default=10,
                    type=int, help="Size of the sampled datasets")
parser.add_argument("--seed", default=7,
                    type=int, help="Random seed used to sample splits to enable reproducibility")

args = parser.parse_args()

VALID_RATIO = 0.25
NUM_FOLDS = 5
DATA_PATH = "data/bach/ICIAR2018_BACH_Challenge/Photos"

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, DATA_PATH)
OUTPUT_PATH = os.path.join(current_dir, "data/bach")
print("DATA PATH", DATA_PATH)

# set seed
# TODO: I manually checked and the sampled images are the same, but it would be neat to add a test here

np.random.seed(args.seed)
print("Random seed:", args.seed)

def get_filepaths_recursively(root_path):
    file_paths, file_labels = [], []
    for cls in os.listdir(root_path):  # for each class
        class_path = os.path.join(root_path, cls)
        if os.path.isdir(class_path): # only loop through folders
            class_imgs = os.listdir(class_path)
            # add file if it's an image
            class_imgs = [os.path.join(class_path, img) for img in class_imgs if img.split(".")[1] == "tif"]
            class_labels = [cls] * len(class_imgs)
            file_paths += class_imgs
            file_labels += class_labels

    file_paths, file_labels = np.array(file_paths), np.array(file_labels)

    return file_paths, file_labels


def save_fold_split(fold_imgs, fold_labels, split, fold):
    # save the filelists
    filelist_imgs_path = os.path.join(OUTPUT_PATH, f"{split}_images_fold{fold}.npy")
    filelist_labels_path = os.path.join(OUTPUT_PATH, f"{split}_labels_fold{fold}.npy")
    np.save(filelist_imgs_path, np.array(fold_imgs))
    np.save(filelist_labels_path, np.array(fold_labels))


imgs, labels = get_filepaths_recursively(DATA_PATH)
# split indices into 5 non-overlapping groups
fold_idxs = np.array_split(np.random.permutation(len(imgs)), NUM_FOLDS)
# create lists to assert correct split of indices
assert_train_idxs = []
assert_val_idxs = []

all_idxs = np.array(range(len(imgs)))
for fold in range(NUM_FOLDS):
    print("Fold ", fold)
    # use the ith group of indices as validation
    val_fold_idxs = fold_idxs[fold]
    print("First 10 val idxs", val_fold_idxs[:10])
    # use everything else as training
    train_fold_idxs = np.setdiff1d(all_idxs, val_fold_idxs)

    train_imgs = imgs[train_fold_idxs]
    train_labels = labels[train_fold_idxs]

    val_imgs = imgs[val_fold_idxs]
    val_labels = labels[val_fold_idxs]

    assert_train_idxs += list(train_fold_idxs)
    assert_val_idxs += list(val_fold_idxs)

    save_fold_split(train_imgs, train_labels, "train", fold)
    save_fold_split(val_imgs, val_labels, "val", fold)

# check that the indices between folds don't overlap
assert len(assert_train_idxs) == (NUM_FOLDS-1) * len(imgs)
assert len(assert_val_idxs) == len(imgs)

# TODO check if the seed makes the split deterministic

