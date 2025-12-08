
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Our Grad-CAM implementation for CNN models.

    Steps:
    - Hook feature maps from target layer
    - Hook gradients during backprop
    - Weight channels by gradient importance
    - Upsample CAM to image size
    - Overlay heatmap on original image
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)


    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor):
        # Forward
        output = self.model(input_tensor)

        # Backward on predicted class
        pred_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[:, pred_class].backward()

        # Global pooling of gradients
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])

        # Weight activations
        cam = (self.activations * pooled_grads[None, :, None, None]).sum(dim=1)

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), pred_class
