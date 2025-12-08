# ================================================================
# Grad-CAM for EfficientNet-B3 (timm-based)
# ================================================================
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM_EfficientNet:
    """
    Grad-CAM for EfficientNet-B3.
    Hooks the final conv layer: model.backbone.conv_head
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # ---- Forward hook: get activations ----
        def fwd_hook(module, inp, out):
            self.activations = out
        
            # register backward hook ON THE TENSOR
            def bwd_hook(grad):
                self.gradients = grad
            out.register_hook(bwd_hook)

        target_layer.register_forward_hook(fwd_hook)

    def __call__(self, input_tensor):
        # Forward pass
        logits = self.model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

        # Backward pass for predicted class
        self.model.zero_grad()
        logits[:, pred_class].backward()

        # Retrieve activations and gradients: shape (1, C, H, W)
        acts = self.activations[0]
        grads = self.gradients[0]

        # Global average pooling of gradients
        weights = grads.mean(dim=[1, 2])  # (C,)

        # Weighted sum of activations
        cam = (weights[:, None, None] * acts).sum(dim=0)

        # Normalize for visualization
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), pred_class

