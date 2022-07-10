import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image
from PIL import Image
import numpy as np
import logging


class NCTDataset(Dataset):
    """
    NCTDataset uses a random split of 100k imag into training set (75%), validation set (25%)
    and a test set containing 7k images. This class is used for the main supervised experiments (TODO add corresponding section name from the final paper).
    It is initialized from a .npy file (generated with create_breakhis_filelist.py) which contains a list of images
    in every split.
    """
    def __init__(self, root_dir, split, old_img_path_prefix=None, new_img_path_prefix=None, transform=None):
        self.label2int = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        imgs_path = os.path.join(root_dir, f"{split}_images.npy")
        labels_path = os.path.join(root_dir, f"{split}_labels.npy")

        logging.info(f"Images path for {split} set: {imgs_path}")

        if (old_img_path_prefix is not None) and (old_img_path_prefix != "None"):
            logging.info(f"Replace image path prefixes from {old_img_path_prefix} to {new_img_path_prefix}")
            self.imgs = [img.replace(old_img_path_prefix, new_img_path_prefix) for img in np.load(imgs_path)]
        else:
            self.imgs = np.load(imgs_path)
        self.labels = [self.label2int[label] for label in np.load(labels_path)]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_label(self, img):
        label_str = img.split('-')[0].split('_')[1]
        return self.label2int[label_str]