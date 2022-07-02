from torch.utils.data import Dataset, DataLoader
import random
import torch


class DummyDataset(Dataset):
    """
        Dummy dataset for testing an demonstration.
        If rotation_invariant=True, it returns an image with all values equal to its index, so the images are not
        affected by rotations, and a random label.
    """

    def __init__(self, img_height=96, img_width=96, num_channels=3, num_classes=2, rotation_invariant=True, size=10):
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.rotation_invariant = rotation_invariant
        self.size = size

    def __getitem__(self, idx):
        if self.rotation_invariant:
            img = torch.ones((self.num_channels, self.img_height, self.img_width)) * idx
        else: # return random images
            img = torch.randn((self.num_channels, self.img_height, self.img_width))

        label = random.randint(0, self.num_classes - 1)

        return img, label

    def __len__(self):
        return self.size