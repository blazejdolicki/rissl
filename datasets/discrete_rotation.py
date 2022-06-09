import torch
from torchvision.transforms.functional import rotate


class DiscreteRotation:
    """Rotate image by one of the given angles.

    Arguments:
        angles: list(ints). List of integer degrees to pick from. E.g. [0, 90, 180, 270] for a random 90-degree-like rotation
    """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = self.angles[torch.randperm(len(self.angles))[0]]
        return rotate(x, angle)

    def __repr__(self):
        return f"{self.__class__.__name__}(angles={self.angles})"
