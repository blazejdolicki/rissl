import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image
from PIL import Image
from datasets.breakhis_dataset import BreakhisDataset


class BreakhisFoldDataset(BreakhisDataset):
    """
    BreakhisFoldDataset contains 5 train/test folds from the original paper such that all images from the same patient
    are in the same fold. This class is used for scale-related experiments (TODO: add name of relevant section in paper).
    It allows selecting specific magnitudes and is initialized by reading image names from folders with the structure
    described below.

    Directory tree:
    <root_dir>/
        fold1/
            train/
                40X/
                    SOB_B_A-14-22549G-40-001.png
                    ...
                100X/
                200X/
                400X/
            test/
        fold2/
        ...
        fold5/
    """
    def __init__(self, root_dir, split, fold, mag, transform=None):
        self.label2int = {"B": 0, "M": 1}
        mag_dir = os.path.join(root_dir, f"fold{fold}", split, f"{mag}X")
        self.imgs = [os.path.join(mag_dir, img) for img in os.listdir(mag_dir)]
        self.labels = [self.get_label(img) for img in os.listdir(mag_dir)]
        self.transform = transform

