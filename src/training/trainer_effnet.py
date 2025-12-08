
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from torch.cuda.amp import autocast, GradScaler

# ------------------------------------------------------------
# MixUp and CutMix utilities
# ------------------------------------------------------------
def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    bbx1 = torch.randint(0, H, (1,)).item()
    bby1 = torch.randint(0, W, (1,)).item()
    cut_h = int(H * torch.sqrt(1 - lam))
    cut_w = int(W * torch.sqrt(1 - lam))

    bbx2 = min(H, bbx1 + cut_h)
    bby2 = min(W, bby1 + cut_w)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# ------------------------------------------------------------
# EfficientNet Trainer
# ------------------------------------------------------------
class EfficientNetTrainer:
    def __init__(self,
                 model,
                 loaders,
                 optimizer,
                 criterion,
                 scheduler=None,
                 device="cuda",
                 use_amp=True,
                 use_mixup=True,
                 use_cutmix=False,
                 alpha=1.0,
                 save_dir="checkpoints/effnet"):

        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp

        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.alpha = alpha

        self.scaler = GradScaler(enabled=use_amp)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = -1
        self.best_epoch = -1

    # ------------------------------------------------------------
    # One epoch of training
    # ------------------------------------------------------------
    def train_one_epoch(self, epoch):
        self.model.train()
        loader = self.loaders["train"]

        running_loss = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # --------------------------------------------
            # Apply augmentation
            # --------------------------------------------
            if self.use_mixup:
                images, y_a, y_b, lam = mixup_data(images, labels, self.alpha)
            elif self.use_cutmix:
                images, y_a, y_b, lam = cutmix_data(images, labels, self.alpha)
            else:
                lam, y_a, y_b = 1.0, labels, labels

            # --------------------------------------------
            # Forward
            # --------------------------------------------
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(outputs, y_b)

            # --------------------------------------------
            # Backward
            # --------------------------------------------
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        return running_loss / total

    # ------------------------------------------------------------
    # One epoch of validation
    # ------------------------------------------------------------
    def validate_one_epoch(self):
        self.model.eval()
        loader = self.loaders["val"]

        total = 0
        correct = 0
        running_loss = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                running_loss += loss.item() * images.size(0)
                total += images.size(0)

        return running_loss / total, correct / total

    # ------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------
    def fit(self, epochs=40):
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate_one_epoch()

            if self.scheduler:
                self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            # Save best checkpoint
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")

        # Save history
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)

        return history
