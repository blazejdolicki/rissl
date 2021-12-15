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

DATA_PATH = "data/nct/"
split_folders = {"train": "NCT-CRC-HE-100K",
                 "test": "CRC-VAL-HE-7K"}

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, DATA_PATH)

print("DATA PATH", DATA_PATH)

# set seed
# TODO: I manually checked and the sampled images are the same, but it would be neat
# to add a test here

random.seed(args.seed)
print("Random seed:", args.seed)

def get_filepaths_recursively(root_path):
    file_paths, file_labels = [], []
    for cls in os.listdir(root_path):  # for each class
        class_path = os.path.join(root_path, cls)
        if os.path.isdir(class_path): # only loop through folders
            class_imgs = os.listdir(class_path)
            class_imgs = [os.path.join(class_path, img) for img in class_imgs]
            class_labels = [cls] * len(class_imgs)
            file_paths += class_imgs
            file_labels += class_labels

    file_paths, file_labels = np.array(file_paths), np.array(file_labels)

    return file_paths, file_labels


def save_split(imgs, labels, split):
    print(f"Saving {split} set with {len(imgs)} examples")
    # save the filelists
    sample_suffix = f"_sample_{args.sample_size}" if args.use_sample else ""
    filelist_imgs_path = os.path.join(DATA_PATH, f"{split}{sample_suffix}_images.npy")
    filelist_labels_path = os.path.join(DATA_PATH, f"{split}{sample_suffix}_labels.npy")
    np.save(filelist_imgs_path, np.array(imgs))
    np.save(filelist_labels_path, np.array(labels))


for split in ["train", "test"]:
    split_path = os.path.join(DATA_PATH, split_folders[split])
    imgs, labels = get_filepaths_recursively(split_path)

    assert len(imgs) == len(labels), "Number of images and labels should be equal"

    if args.use_sample:
        print(f"Using sample of {split} split with {args.sample_size} examples")
        # sample k indices
        random_idxs = random.sample(range(len(imgs)), args.sample_size)
        # sample images and labels with those indices
        imgs = imgs[random_idxs]
        labels = labels[random_idxs]

    if split == "train":
        # create and save validation set
        all_idxs = np.array(range(len(imgs)))
        val_idxs = random.sample(list(all_idxs), int(len(imgs) * VALID_RATIO))
        train_idxs = np.setdiff1d(all_idxs, val_idxs)

        assert len(train_idxs) + len(val_idxs) == len(all_idxs)
        assert len(np.intersect1d(train_idxs, val_idxs)) == 0, \
            "Image indices shouldn't overlap between train and validation sets"

        val_imgs = imgs[val_idxs]
        val_labels = labels[val_idxs]
        save_split(val_imgs, val_labels, "valid")

        imgs = imgs[train_idxs]
        labels = labels[train_idxs]

    save_split(imgs, labels, split)



