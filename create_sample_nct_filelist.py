import os
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_sample", action="store_true", help="Use sample of the dataset.")
parser.add_argument("--sample_size", default=10,
                    type=int, help="Size of the sampled datasets")
args = parser.parse_args()

DATA_PATH = "data/nct/"
split_folders = {"train": "NCT-CRC-HE-100K",
                 "test": "CRC-VAL-HE-7K"}

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, DATA_PATH)

print("DATA PATH", DATA_PATH)

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


for split in ["train", "test"]:
    split_path = os.path.join(DATA_PATH, split_folders[split])
    imgs, labels = get_filepaths_recursively(split_path)

    assert len(imgs) == len(labels), "Number of images and labels should be equal"

    print(f"Using full {split} split with {len(imgs)} examples")
    if args.use_sample:
        print(f"Using sample of {split} split with {args.sample_size} examples")
        # sample k indices
        random_idxs = random.sample(range(len(imgs)), args.sample_size)
        # sample images and labels with those indices
        imgs = imgs[random_idxs]
        labels = labels[random_idxs]

    # TODO: Currently we don't shuffle the full dataset, maybe we want to do it in the future

    # save the filelists
    sample_suffix = f"_sample_{args.sample_size}" if args.use_sample else ""
    filelist_imgs_path = os.path.join(DATA_PATH, f"{split}{sample_suffix}_images.npy")
    filelist_labels_path = os.path.join(DATA_PATH, f"{split}{sample_suffix}_labels.npy")
    np.save(filelist_imgs_path, np.array(imgs))
    np.save(filelist_labels_path, np.array(labels))


