
"""
ISIC 2018 Dataset Loader (Metadata-Driven)
=========================================

This module defines our PyTorch Dataset class for the ISIC-2018
skin cancer image classification task.

It loads the unified metadata CSV file, applies transforms, and
returns (image_tensor, label) pairs compatible with deep learning
training on GPUs.

Usage:
    from data.dataset_isic import ISICDataset
    from data.transforms import TransformFactory

    train_ds = ISICDataset(
        metadata_csv="data/processed/isic2018_metadata.csv",
        split="train",
        image_size=224
    )

"""

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from data.transforms import TransformFactory


class ISICDataset(Dataset):
    """
    Our metadata-driven PyTorch Dataset for ISIC-2018.

    Parameters
    ----------
    metadata_csv : str or Path
        Path to the unified metadata CSV file.
    split : str
        One of ["train", "val", "test"].
    image_size : int
        Target image resolution (default 224).
    use_imagenet : bool
        If True, apply ImageNet normalization for pretrained models.
        If False, use ISIC-specific normalization.
    transform_override : torchvision.transforms.Compose or None
        If provided, overrides default transforms from TransformFactory.
    """

    def __init__(self,
                 metadata_csv,
                 split: str = "train",
                 image_size: int = 224,
                 use_imagenet: bool = False,
                 transform_override=None):

        self.metadata_path = Path(metadata_csv)
        self.split = split
        self.image_size = int(image_size)
        self.use_imagenet = bool(use_imagenet)

        # Load metadata
        df = pd.read_csv(self.metadata_path)

        # Filter for correct split
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Choose transforms
        if transform_override is not None:
            self.transforms = transform_override
        else:
            self.transforms = TransformFactory.get_transforms(
                split=split,
                image_size=image_size,
                use_imagenet=use_imagenet
            )

    def __len__(self):
        """
        Total number of samples in dataset split.
        """
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Fetch one sample (image + label).

        Returns
        -------
        image_tensor : torch.Tensor
            The transformed image.
        label : int
            Integer class label (0–6).
        """

        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, label

