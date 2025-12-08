
"""
Training engine for our Hybrid Scratch CNN on ISIC-2018.
=======================================================

This module provides a reusable training loop with:
- Mixed precision (optional)
- Multi-GPU support (DataParallel)
- Checkpointing (best + last + periodic)
- History logging (loss/accuracy per epoch)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import json
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    """
    Our configuration dataclass for training.

    Parameters
    ----------
    epochs : int
        Total number of epochs to train.
    device : str
        Device string, e.g., "cuda" or "cpu".
    amp : bool
        Whether to use automatic mixed precision.
    checkpoint_dir : str or Path
        Directory where we store checkpoints.
    history_path : str or Path
        File path for saving training history JSON.
    best_metric : str
        Metric to decide the "best" model. One of ["val_loss", "val_acc"].
    periodic_ckpt_freq : int
        Frequency (in epochs) at which we save periodic checkpoints.
    """

    epochs: int = 50
    device: str = "cuda"
    amp: bool = True
    checkpoint_dir: str = "checkpoints/scratch_model"
    history_path: str = "logs/training/scratch_model_history.json"
    best_metric: str = "val_loss"  # or "val_acc"
    periodic_ckpt_freq: int = 10


class ScratchTrainer:
    """
    Our trainer class for the Hybrid Scratch CNN baseline.

    We encapsulate:
    - Model
    - DataLoaders
    - Loss function
    - Optimizer
    - Scheduler (optional)
    - AMP scaler (if enabled)
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        criterion: nn.Module,
        optimizer: Optimizer,
        config: TrainConfig,
        scheduler: Optional[_LRScheduler] = None,
    ):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Prepare paths
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = Path(config.history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # DataParallel if multiple GPUs and device is CUDA
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        # AMP scaler
        self.use_amp = config.amp and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Training history
        self.history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        # Tracking best model
        self.best_metric_value = None
        self.best_epoch = None

    # ---------------------------------------------------
    # Core training API
    # ---------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop for the configured number of epochs.

        Returns
        -------
        dict
            Training history with loss/accuracy per epoch.
        """
        print("Starting training for", self.config.epochs, "epochs.")

        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")

            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._eval_one_epoch(epoch)

            # Scheduler step (if any) based on validation
            if self.scheduler is not None:
                # Most schedulers step per epoch; some use val_loss
                try:
                    # For schedulers like ReduceLROnPlateau (requires metric)
                    self.scheduler.step(val_loss)
                except TypeError:
                    # For schedulers like OneCycleLR / CosineAnnealing (no metric)
                    self.scheduler.step()

            # Log metrics
            current_lr = self._get_current_lr()
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            print(
                f"Epoch {epoch} Summary - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Checkpointing
            self._maybe_save_checkpoint(epoch, val_loss, val_acc)

            # Save history to disk after each epoch
            self._save_history()

        print("\nTraining completed.")
        print("Best epoch:", self.best_epoch)
        print("Best metric value:", self.best_metric_value)

        return self.history

    # ---------------------------------------------------
    # Internal: one training epoch
    # ---------------------------------------------------
    def _train_one_epoch(self, epoch: int):
        """
        One epoch of supervised training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loader = self.loaders["train"]

        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Compute batch accuracy
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 50 == 0:
                print(
                    f"  [Train] Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    # ---------------------------------------------------
    # Internal: one evaluation epoch
    # ---------------------------------------------------
    def _eval_one_epoch(self, epoch: int):
        """
        One epoch of evaluation on validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        loader = self.loaders["val"]

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return epoch_loss, epoch_acc

    # ---------------------------------------------------
    # Internal: learning rate helper
    # ---------------------------------------------------
    def _get_current_lr(self) -> float:
        """
        Retrieve current learning rate from optimizer.
        """
        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        if len(lrs) == 0:
            return 0.0
        return lrs[0]

    # ---------------------------------------------------
    # Internal: checkpointing
    # ---------------------------------------------------
    def _maybe_save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """
        Save checkpoints based on best metric and periodic frequency.
        """
        # Decide metric to track
        if self.config.best_metric == "val_acc":
            current_metric = val_acc
            better = (
                self.best_metric_value is None or current_metric > self.best_metric_value
            )
        else:  # "val_loss"
            current_metric = val_loss
            better = (
                self.best_metric_value is None or current_metric < self.best_metric_value
            )

        # Save best checkpoint
        if better:
            self.best_metric_value = current_metric
            self.best_epoch = epoch
            self._save_checkpoint(epoch, is_best=True)

        # Save periodic checkpoint
        if self.config.periodic_ckpt_freq > 0 and epoch % self.config.periodic_ckpt_freq == 0:
            self._save_checkpoint(epoch, is_best=False)

        # Always save "last" checkpoint
        self._save_checkpoint(epoch, is_best=False, last=True)

    def _save_checkpoint(self, epoch: int, is_best: bool = False, last: bool = False):
        """
        Save a checkpoint to disk.

        We store:
        - model state_dict
        - optimizer state_dict
        - scheduler state_dict (if any)
        - scaler state_dict (if AMP enabled)
        - current epoch
        """
        state = {
            "epoch": epoch,
            "model_state": self._get_model_state(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "config": asdict(self.config),
        }

        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()

        if self.use_amp:
            state["scaler_state"] = self.scaler.state_dict()

        # Base filename
        ckpt_name = f"epoch_{epoch}.pth"

        if is_best:
            ckpt_name = "best.pth"
        if last:
            ckpt_name = "last.pth"

        ckpt_path = self.checkpoint_dir / ckpt_name
        torch.save(state, ckpt_path)
        tag = "BEST" if is_best else ("LAST" if last else "CKPT")
        print(f"  [{tag}] Saved checkpoint at: {ckpt_path}")

    def _get_model_state(self):
        """
        Retrieve state_dict from model, handling DataParallel.
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    # ---------------------------------------------------
    # Internal: history saving
    # ---------------------------------------------------
    def _save_history(self):
        """
        Save training history to our JSON log file.
        """
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=4)
