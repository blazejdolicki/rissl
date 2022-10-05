import os
import random
import numpy as np
import argparse




# root_path here is /mnt/archive/projectdata/data_pcam/disk_folder/train
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


def save_split(imgs, labels, split, data_path, sample_size):
    print(f"Saving {split} set with {len(imgs)} examples")
    # save the filelists
    filelist_imgs_path = os.path.join(data_path, f"{split}_sample_{sample_size}_images.npy")
    filelist_labels_path = os.path.join(data_path, f"{split}_sample_{sample_size}_labels.npy")
    np.save(filelist_imgs_path, np.array(imgs))
    np.save(filelist_labels_path, np.array(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=float, help="Size of the sampled datasets")
    parser.add_argument("--seed", default=7,
                        type=int, help="Random seed used to sample splits to enable reproducibility")
    parser.add_argument("--data_dir", type=str, help="Directory where the data is stored and where"
                                                     "the output files will be store")
    args = parser.parse_args()

    random.seed(args.seed)
    print("Random seed:", args.seed)
    split_path = os.path.join(args.data_dir, "train")
    imgs, labels = get_filepaths_recursively(split_path)

    assert len(imgs) == len(labels), "Number of images and labels should be equal"

    sample_size = int(args.sample_size*len(imgs))
    print(f"Sampling {sample_size} out of {len(imgs)} examples")
    # sample k indices
    random_idxs = random.sample(range(len(imgs)), sample_size)
    # sample images and labels with those indices
    imgs = imgs[random_idxs]
    labels = labels[random_idxs]

    save_split(imgs, labels, "train", args.data_dir, args.sample_size)