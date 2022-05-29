import torch
from torch.utils.data import Dataset
import os
import h5py
from PIL import Image
import random


class PCamDataset(Dataset):
    """
    PCam dataset is derived from the Camelyon16 challenge.
    It consists of 327,680 color images (train - 262,144, validation - 32.768, test - 32.768) with a binary label
    indicating presence of metastatic tissue. The images are relatively small (96 x 96 px).
    A positive label indicates that the center 32x32 px region of a patch contains at least one pixel of tumor tissue.
    Tumour tissue in the outer region does not influence the label.

    Source: https://github.com/basveeling/pcam

    :param sample: Allows to take a sample of the dataset.
                   If `sample` is a fraction between 0 and 1, we take a proportional fraction of the full dataset.
                   If `sample` is a whole number, we take a number of elements from the full dataset equal to `sample`.
    """

    def __init__(self, root_dir, split, transform=None, sample=None):
        self.sample = sample

        imgs_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{split}_x.h5')
        self.imgs = h5py.File(imgs_path, 'r')["x"]

        labels_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{split}_y.h5')
        self.labels = h5py.File(labels_path, 'r')["y"]

        if sample is not None:
            self.get_sample()

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx].item()
        # convert to PIL image as required by Resize (old Pytorch version)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_sample(self):
        dataset_size = len(self)
        if self.sample < 1:
            self.sample = self.sample * dataset_size
        # h5py requires the indices to be sorted in increasing order
        sample_idxs = sorted(random.sample(range(dataset_size), int(self.sample)))

        self.imgs = self.imgs[sample_idxs]
        self.labels = self.labels[sample_idxs]
