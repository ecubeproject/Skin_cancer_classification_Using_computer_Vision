
"""
Transform Pipeline for ISIC-2018 Skin Cancer Classification
===========================================================

This module defines dataset transforms for:

1. Training
2. Validation
3. Test

It supports:
- ISIC-specific normalization (derived from dataset pixel statistics)
- ImageNet normalization (for pretrained CNNs)
- Strong augmentation for training
- Deterministic transforms for validation/test

Usage:
    from data.transforms import TransformFactory
    train_tf = TransformFactory.get_transforms("train", use_imagenet=False)
"""

import torchvision.transforms as T


# ============================================================
# Normalization Constants
# ============================================================

# ISIC-specific normalization (from EDA pixel statistics)
ISIC_MEAN = [0.7635211992481025, 0.5461279559977089, 0.5705304120840731]
ISIC_STD = [0.14121190060202815, 0.15289106131799016, 0.17032798674104263]

# ImageNet normalization (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Transform Factory
# ============================================================

class TransformFactory:
    """
    Our factory class for creating train/val/test transforms.

    Parameters
    ----------
    split : str
        One of ["train", "val", "test"].
    image_size : int
        The input resolution for the model (default 224).
    use_imagenet : bool
        If True, uses ImageNet normalization (for pretrained models).
        If False, uses ISIC-specific normalization.
    """

    @staticmethod
    def get_transforms(split: str,
                       image_size: int = 224,
                       use_imagenet: bool = False):

        # Choose normalization
        if use_imagenet:
            norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else:
            norm = T.Normalize(mean=ISIC_MEAN, std=ISIC_STD)

        # ----------------------------
        # TRAIN TRANSFORMS
        # ----------------------------
        if split == "train":
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=25),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02
                ),
                T.RandomResizedCrop(
                    size=image_size,
                    scale=(0.80, 1.0),
                    ratio=(0.90, 1.10)
                ),
                T.ToTensor(),
                norm,
            ])

        # ----------------------------
        # VAL / TEST TRANSFORMS
        # ----------------------------
        elif split in ["val", "test"]:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                norm,
            ])

        else:
            raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'val', 'test'].")

