
"""
EfficientNet-B3 (timm version) for ISIC-2018
============================================

This module replaces the torchvision implementation because:
- torchvision EfficientNet-B3 weights cannot be downloaded on some networks
- CDN hash checks fail frequently (403 or corrupted file)
- timm weights load reliably and provide equal or better accuracy

We remove the original classifier head and replace it with a
7-class classification head suitable for ISIC-2018.
"""

import torch
import torch.nn as nn
import timm


class ISICEfficientNetB3(nn.Module):
    """
    EfficientNet-B3 backbone with custom classifier head for ISIC-2018.

    Parameters
    ----------
    num_classes : int
        Number of lesion classes (ISIC-2018 = 7).
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    dropout_p : float
        Dropout rate before final classifier.
    freeze_backbone : bool
        If True, freezes EfficientNet layers for feature extraction.
    """

    def __init__(self, num_classes=7, pretrained=True, dropout_p=0.4, freeze_backbone=False):
        super().__init__()

        # Load EfficientNet-B3 from timm
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,         # remove classifier
            global_pool="avg"
        )

        # Custom head
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

        # Optional freezing
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)   # (B, C)
        x = self.dropout(features)
        x = self.fc(x)
        return x

    def count_parameters(self):
        """
        Count trainable parameters only.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Simple test
    model = ISICEfficientNetB3(num_classes=7, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Trainable parameters:", model.count_parameters())
