import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image
from PIL import Image

class BreakhisDataset(Dataset):
    """
    BreakHis includes 7909 patches of size 700 ×460 pixels taken from WSIs of breast tumor tissue.
    The data is labelled as either benign or malignant, and images belong to one of four magnifying
    factors (40x, 100x, 200x and 400x).
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