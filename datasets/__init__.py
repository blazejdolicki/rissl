import logging

from torchvision import transforms
import os
from enum import Enum

from datasets.pcam_dataset import PCamDataset
from datasets.breakhis_dataset import BreakhisDataset
from datasets.breakhis_fold_dataset import BreakhisFoldDataset
from datasets.discrete_rotation import DiscreteRotation
from models import is_equivariant
from enum import Enum




def get_transforms(args):
    # define img sizes for which all feature maps are odd-sized (required for equivariance)
    e2_img_sizes = {"breakhis": 449,
                     "pcam": 97
                     }
    # define img size of all datasets
    data_img_sizes = {"pcam": 96}

    data_transforms = {
                        "_default": {
                            "train": [
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], # statistics from ImageNet
                                                     std=[0.229, 0.224, 0.225])
                            ],
                            "test": [
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], # statistics from ImageNet
                                                     std=[0.229, 0.224, 0.225])
                            ]
                        },
                        "pcam": {
                            "train": [
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                DiscreteRotation(angles=[0, 90, 180, 270]),
                                transforms.ColorJitter(brightness=64/255,
                                                       saturation=0.25,
                                                       hue=0.04,
                                                       contrast=0.75),
                                #  8px jitter (shift) following Liu et al 2017
                                transforms.RandomAffine(degrees=0,
                                                        translate=(4/data_img_sizes["pcam"],
                                                                   4/data_img_sizes["pcam"])),
                                transforms.ToTensor(),
                            ],
                            "test": [transforms.ToTensor()]

                        },
                        "breakhis": {
                            "train": [
                                transforms.CenterCrop(460 if not is_equivariant(args.model_type) else e2_img_sizes["breakhis"]),
                                DiscreteRotation(angles=[0, 90, 180, 270]),
                                transforms.ToTensor()
                            ],
                            "test": [
                                transforms.CenterCrop(460 if not is_equivariant(args.model_type) else e2_img_sizes["breakhis"]),
                                transforms.ToTensor()
                            ]
                        },
    }
    if args.dataset not in data_transforms:
        logging.warning("The specified dataset does not have custom transforms, using default transfroms")
        args.dataset = "_default"

    train_transform_list = data_transforms[args.dataset]["train"]
    test_transform_list = data_transforms[args.dataset]["test"]

    # To preserve equivariance, we need to pick image size that ensures all feature maps are odd-sized
    # https://arxiv.org/pdf/2004.09691.pdf (Figure 2)
    if is_equivariant(args.model_type):
        resize = transforms.Resize(e2_img_sizes[args.dataset])
        train_transform_list.insert(0, resize)
        test_transform_list.insert(0, resize)

    # remove rotation transformations
    if args.no_rotation_transforms:
        logging.info("Not using rotation transforms")
        remove_rotation_transforms(train_transform_list)

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)

    return train_transform, test_transform


def remove_rotation_transforms(transform_list):
    for t in transform_list:
        if isinstance(t, DiscreteRotation):
            transform_list.remove(t)

def get_dataset(train_transform, test_transform, args):
    if args.sample is not None and args.dataset != "pcam":
        raise NotImplementedError("Currently sampling is only implement for PCam")

    if args.dataset == "breakhis_fold":
        breakhis_dir = os.path.join(args.data_dir, "breakhis")
        train_dataset = BreakhisFoldDataset(breakhis_dir, "train", args.fold, args.train_mag, train_transform)
        test_dataset = BreakhisFoldDataset(breakhis_dir, "val", args.fold, args.test_mag, test_transform)
    elif args.dataset == "breakhis":
        num_classes = 2
        train_dataset = BreakhisDataset(args.data_dir, "train", args.old_img_path_prefix, args.new_img_path_prefix,
                                        train_transform)
        test_dataset = BreakhisDataset(args.data_dir, "val", args.old_img_path_prefix, args.new_img_path_prefix,
                                       test_transform)
    elif args.dataset == "pcam":
        num_classes = 2
        train_dataset = PCamDataset(root_dir=args.data_dir, split="train", transform=train_transform, sample=args.sample)
        test_dataset = PCamDataset(root_dir=args.data_dir, split="valid", transform=test_transform, sample=args.sample)

    return train_dataset, test_dataset, num_classes


def convert_transform_to_dict(transform):
    transform_vars = {}
    transform_vars["name"] = transform.__class__.__name__

    for var, value in transform.__dict__.items():
        # Enums cannot be serialized to JSON
        if var != "training" and not var.startswith("_") and not isinstance(value, Enum):
            transform_vars[var] = value
    return transform_vars


def convert_dict_to_transform(transform_dict):
    transform_args = {k: v for k, v in transform_dict.items() if k != "name"}
    transform = getattr(transforms, transform_dict["name"])(**transform_args)
    return transform