
"""
High-Performance DataLoaders for ISIC-2018
=========================================

This module provides optimized DataLoader constructors for
training, validation, and testing on ISIC-2018. It integrates:

- Metadata-driven ISICDataset
- WeightedRandomSampler for class imbalance correction
- Multi-worker data loading (num_workers=8)
- GPU-optimized configuration (pin_memory, persistent workers)
"""

import torch
from torch.utils.data import DataLoader
from data.dataset_isic import ISICDataset
from data.sampler_isic import create_weighted_sampler


def get_dataloaders(metadata_csv,
                    image_size=224,
                    batch_size=32,
                    use_imagenet=False,
                    num_workers=8,
                    prefetch_factor=4):
    """
    Create train, val, and test DataLoaders for ISIC-2018.

    Parameters
    ----------
    metadata_csv : str or Path
        Path to unified metadata CSV.
    image_size : int
        Input image resolution.
    batch_size : int
        Batch size for all loaders.
    use_imagenet : bool
        Whether to use ImageNet normalization (for pretrained models).
    num_workers : int
        Number of worker threads for data loading.
    prefetch_factor : int
        DataLoader prefetch factor.

    Returns
    -------
    dict
        Dictionary with train_loader, val_loader, test_loader.
    """

    # ---------------------------
    # Create dataset objects
    # ---------------------------
    train_ds = ISICDataset(
        metadata_csv=metadata_csv,
        split="train",
        image_size=image_size,
        use_imagenet=use_imagenet
    )

    val_ds = ISICDataset(
        metadata_csv=metadata_csv,
        split="val",
        image_size=image_size,
        use_imagenet=use_imagenet
    )

    test_ds = ISICDataset(
        metadata_csv=metadata_csv,
        split="test",
        image_size=image_size,
        use_imagenet=use_imagenet
    )

    # ---------------------------
    # Balanced sampler for train
    # ---------------------------
    train_sampler = create_weighted_sampler(train_ds)

    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        sampler=train_sampler,        # Balanced sampling
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
