
"""
Balanced Sampling Utilities for ISIC-2018
=========================================

This module defines helper functions for creating balanced
WeightedRandomSampler instances for the ISICDataset.

The sampler corrects extreme class imbalance, improves recall
for minority classes, and is compatible with DataLoader and
DistributedDataParallel (DDP).

Usage:
    from data.sampler_isic import create_weighted_sampler
    sampler = create_weighted_sampler(train_dataset)
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_sample_weights(dataset):
    """
    Compute a per-sample weight array based on class frequencies.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        An ISICDataset instance for the training split.

    Returns
    -------
    np.ndarray
        Array of per-sample weights.
    """

    # Extract labels from the metadata-driven dataset
    labels = [dataset.df.iloc[i]["label"] for i in range(len(dataset))]

    labels = np.array(labels)
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Compute class frequency
    class_freq = class_counts / class_counts.sum()

    # Compute class weights (inverse frequency)
    class_weights = 1.0 / (class_freq + 1e-8)

    # Normalize class weights
    class_weights = class_weights / class_weights.sum()

    # Assign weight to each sample based on its class
    sample_weights = class_weights[labels]

    return sample_weights


def create_weighted_sampler(dataset):
    """
    Create a WeightedRandomSampler object from a given dataset.

    Parameters
    ----------
    dataset : ISICDataset
        The training dataset instance.

    Returns
    -------
    WeightedRandomSampler
        A PyTorch sampler suitable for DataLoader.
    """

    sample_weights = compute_sample_weights(dataset)

    # Convert to torch tensor
    weight_tensor = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(weight_tensor),  # sample entire dataset each epoch
        replacement=True                 # needed for oversampling
    )

    return sampler
