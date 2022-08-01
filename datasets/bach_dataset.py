import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image
from PIL import Image
import numpy as np
import logging

class BachDataset(Dataset):
    """

    """
    def __init__(self, root_dir, split, fold=None, old_img_path_prefix=None, new_img_path_prefix=None, transform=None):
        self.label2int = {"Benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}

        assert (fold is not None) or split != "val", "Specify fold unless you're using test set"

        if (fold is not None) and (fold != "None"):
            imgs_path = os.path.join(root_dir, f"{split}_images_fold{fold}.npy")
            labels_path = os.path.join(root_dir, f"{split}_labels_fold{fold}.npy")
        else:
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