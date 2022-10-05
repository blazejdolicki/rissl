import os
import random
import numpy as np
import argparse

"""
We first split the whole dataset into a set used for cross validation and a separate test set.
Then we split the first set into k samples and from them we sample training and valdiation set for every epoch.
"""

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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=7,
                    type=int, help="Random seed used to sample splits to enable reproducibility")
parser.add_argument("--data_dir", type=str, help="Directory where the data is stored and where"
                                                 "the output files will be store")
args = parser.parse_args()

CV_RATIO = 0.9
TEST_RATIO = 0.1

NUM_FOLDS = 5
# path relative to bach folder
DATA_PATH = "ICIAR2018_BACH_Challenge/Photos"
DATA_PATH = os.path.join(args.data_dir, DATA_PATH)
OUTPUT_PATH = args.data_dir
print("DATA PATH", DATA_PATH)

# set seed
np.random.seed(args.seed)
print("Random seed:", args.seed)

imgs, labels = get_filepaths_recursively(DATA_PATH)

# 1. Split the whole dataset into training set and test set

# shuffle all image indices
random_idxs = np.random.permutation(len(imgs))
# split shuffled indices into 2 lists based on the train-val-test ratios
num_cv_examples = int(CV_RATIO*len(imgs))
split_points = [num_cv_examples]
cv_test_idxs = np.split(random_idxs, split_points)

# These idxs are sampled from range [0,len(imgs)]
cv_idxs = cv_test_idxs[0]
test_idxs = cv_test_idxs[1]

cv_imgs = imgs[cv_idxs]
cv_labels = labels[cv_idxs]

# save cv filelist
print("All training data")
print("# training examples", len(cv_imgs))
test_imgs_path = os.path.join(OUTPUT_PATH, f"train_images.npy")
test_labels_path = os.path.join(OUTPUT_PATH, f"train_labels.npy")
np.save(test_imgs_path, np.array(cv_imgs))
np.save(test_labels_path, np.array(cv_labels))


test_imgs = imgs[test_idxs]
test_labels = labels[test_idxs]

# save test filelist
test_imgs_path = os.path.join(OUTPUT_PATH, f"test_images.npy")
test_labels_path = os.path.join(OUTPUT_PATH, f"test_labels.npy")
np.save(test_imgs_path, np.array(test_imgs))
np.save(test_labels_path, np.array(test_labels))

# 2. Within the cross-validation set, split indices into 5 non-overlapping groups

fold_idxs = np.array_split(np.random.permutation(len(cv_imgs)), NUM_FOLDS)
# create lists to assert correct split of indices
assert_train_idxs = []
assert_val_idxs = []

# These idxs are sampled from range [0,len(cv_imgs)]
cv_idxs = np.array(range(len(cv_imgs)))
for fold in range(NUM_FOLDS):
    print("Fold ", fold)
    # use the ith group of indices as validation
    val_fold_idxs = fold_idxs[fold]
    # use everything else as training
    train_fold_idxs = np.setdiff1d(cv_idxs, val_fold_idxs)

    train_imgs = cv_imgs[train_fold_idxs]
    train_labels = cv_labels[train_fold_idxs]

    val_imgs = cv_imgs[val_fold_idxs]
    val_labels = cv_labels[val_fold_idxs]

    # print("First 5 examples from training set", train_imgs[:5])
    # print("First 5 examples from validation set", val_imgs[:5])

    assert len(np.intersect1d(train_imgs, val_imgs)) == 0, \
        "Some images are both in training and validation set of the fold"

    assert len(np.intersect1d(train_imgs, test_imgs)) == 0, \
        "Some images are both in training set of the fold and test set"

    assert len(np.intersect1d(val_imgs, test_imgs)) == 0, \
        "Some images are both in validation set of the fold and test set"

    assert_train_idxs += list(train_fold_idxs)
    assert_val_idxs += list(val_fold_idxs)

    save_fold_split(train_imgs, train_labels, "train", fold)
    save_fold_split(val_imgs, val_labels, "val", fold)

    print(f"# training examples: {len(train_imgs)}, # validation examples {len(val_imgs)}")

print(f"# test examples: {len(test_imgs)}")
# check that the indices between folds don't overlap
assert len(assert_train_idxs) == (NUM_FOLDS-1) * len(cv_imgs)
assert len(assert_val_idxs) == len(cv_imgs)

