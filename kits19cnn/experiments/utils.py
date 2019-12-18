import os
import random
import json
import numpy as np
import torch
import albumentations as albu
from copy import deepcopy

from kits19cnn.io import CenterCrop

def get_training_augmentation(augmentation_key="aug1"):
    transform_dict = {
        "resunet1": [
            albu.HorizontalFlip(p=0.5),
        ],
        "resunet2": [
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=30, interpolation=1, border_mode=4, value=None,
                        mask_value=None, always_apply=False, p=1),
        ],
        "resnet": [
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=30, interpolation=1, border_mode=4, value=None,
                        mask_value=None, always_apply=False, p=1),
            albu.Compose([
                albu.RandomScale(scale_limit=0.1, p=1),
                CenterCrop(height=256, width=256, p=1),
            ], p=0.66)
        ]
    }
    train_transform = transform_dict[augmentation_key]
    print(f"Train Transforms: {train_transform}")
    return albu.Compose(train_transform)

def get_validation_augmentation(augmentation_key):
    """
    Validation data augmentations. Usually, just cropping.
    """
    transform_dict = {
                        "default": [],
                     }
    test_transform = transform_dict[augmentation_key]
    print(f"\nTest/Validation Transforms: {test_transform}")
    return albu.Compose(test_transform)

def get_preprocessing(rgb: bool = False):
    """
    Construct preprocessing transform

    Args:
        rgb (bool): Whether or not to return the input with three channels
            or just single (grayscale)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.pytorch.ToTensorToTensorV2(),
    ]

    print(f"\nPreprocessing Transforms: {_transform}")
    return albu.Compose(_transform)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
