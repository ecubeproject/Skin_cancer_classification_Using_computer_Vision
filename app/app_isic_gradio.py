"""
Gradio app for ISIC-2018 skin lesion classification.

- EfficientNet-B3 (better model) is used for:
  * 7-class prediction (MEL, NV, BCC, AKIEC, BKL, DF, VASC)
  * Malignant vs Benign probability

- Scratch CNN is used ONLY for Grad-CAM:
  * Grad-CAM heatmap
  * Overlay of heatmap on input image

Save this file in:
    /home/ecube/sandiego/AAI_521/module_7/Final_project/app_isic_gradio.py

Then run:
    conda activate AAI521_mod4   # or your env
    cd /home/ecube/sandiego/AAI_521/module_7/Final_project
    python app_isic_gradio.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn.functional as F
import gradio as gr

# ------------------------------------------------------------------
# 1. Project paths and imports from your src/ package
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.models.model_effnet import ISICEfficientNetB3
from src.models.model_scratch import ISICScratchCNN
from src.data.transforms import TransformFactory  # uses ISIC / ImageNet norms

# ------------------------------------------------------------------
# 2. Constants: class labels and malignant/benign mapping
# ------------------------------------------------------------------
CLASS_LABELS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_LABELS)}

# From your notebook: malignant = MEL, BCC; others benign
MALIGNANT_LABELS = ["MEL", "BCC"]
MALIGNANT_INDICES = [CLASS_TO_INDEX[c] for c in MALIGNANT_LABELS]

# Short explanation text (~40 words) for Grad-CAM
GRADCAM_EXPLANATION = (
    "Grad-CAM highlights the image regions that drive the scratch CNN’s prediction. "
    "By overlaying heatmaps on dermoscopy images, clinicians can verify that the model "
    "attends to clinically meaningful structures instead of artifacts, improving trust, "
    "error analysis, safety in real-world decisions."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 3. Load models and checkpoints
# ------------------------------------------------------------------
EFFNET_CKPT = PROJECT_ROOT / "checkpoints" / "effnet_b3_baseline" / "best.pth"
SCRATCH_CKPT = PROJECT_ROOT / "checkpoints" / "scratch_cnn_baseline" / "best.pth"
# If your scratch checkpoint file name is different (e.g. "last.pth"),
# just change SCRATCH_CKPT above.


def load_model_effnet():
    model = ISICEfficientNetB3(
        num_classes=7,
        pretrained=False,   # we load our fine-tuned weights, so no need to pull from timm again
        dropout_p=0.4,
        freeze_backbone=False,
    )
    state = torch.load(EFFNET_CKPT, map_location=DEVICE)
    # Trainer saved state["model_state"]
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_model_scratch():
    model = ISICScratchCNN(num_classes=7)
    state = torch.load(SCRATCH_CKPT, map_location=DEVICE)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


effnet_model = load_model_effnet()
scratch_model = load_model_scratch()

# ------------------------------------------------------------------
# 4. Transforms for inference
#    - EfficientNet: ImageNet normalization (use_imagenet=True)
#    - Scratch CNN: ISIC-specific normalization (use_imagenet=False)
# ------------------------------------------------------------------
EFFNET_TRANSFORM = TransformFactory.get_transforms(
    split="test", image_size=224, use_imagenet=True
)
SCRATCH_TRANSFORM = TransformFactory.get_transforms(
    split="test", image_size=224, use_imagenet=False
)

# ------------------------------------------------------------------
# 5. Grad-CAM implementation FOR SCRATCH CNN ONLY
# ------------------------------------------------------------------


class GradCAMScratch:
    """
    Simple Grad-CAM for the scratch CNN.

    We use the output of stage4[1] (last convolutional block) as the target layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None  # feature maps
        self.gradients = None    # gradients w.r.t. feature maps

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output shape: (B, C, H, W)
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (B, C, H, W)
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # use full_backward_hook for modern PyTorch
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        input_tensor: torch.Tensor of shape (1, 3, H, W), already normalized.
        class_idx: target class for which to compute Grad-CAM.
                   If None, uses model's predicted class.
        Returns:
            cam: numpy array (H, W) with values in [0, 1]
        """
        self.model.zero_grad()
        # Forward pass
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Scalar score for the target class
        score = logits[:, class_idx]
        score.backward()

        # Grab gradients and activations
        grads = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global-average-pool the gradients over spatial dims
        weights = grads.mean(dim=(1, 2))   # (C,)

        # Weighted combination of channels
        cam = torch.zeros_like(activations[0])
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()


# Instantiate Grad-CAM using the last conv block of stage4
gradcam_scratch = GradCAMScratch(
    model=scratch_model,
    target_layer=scratch_model.stage4[1],
)

# ------------------------------------------------------------------
# 6. Helper: prediction + Grad-CAM pipeline
# ------------------------------------------------------------------


def analyze_image(image_path: str):
    """
    Main function called by Gradio:
       - loads image
       - EfficientNet prediction (malignant vs benign + 7-class)
       - Grad-CAM using scratch CNN
       - returns visualizations + text
    """

    if image_path is None or image_path == "":
        return (
            None,
            "Please upload a dermoscopy image first.",
            "",
            None,
            None,
            GRADCAM_EXPLANATION,
        )

    # --------------------------------------------------------------
    # Load and resize image to 224x224 (same as training)
    # --------------------------------------------------------------
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized)  # RGB, uint8

    # --------------------------------------------------------------
    # EfficientNet-B3 prediction
    # --------------------------------------------------------------
    with torch.no_grad():
        x_eff = EFFNET_TRANSFORM(img_resized).unsqueeze(0).to(DEVICE)
        logits_eff = effnet_model(x_eff)
        probs_eff = F.softmax(logits_eff, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs_eff))
    top_prob = float(probs_eff[top_idx])
    top_label = CLASS_LABELS[top_idx]

    malignant_prob = float(probs_eff[MALIGNANT_INDICES].sum())
    benign_prob = 1.0 - malignant_prob

    if malignant_prob >= benign_prob:
        binary_label = "Malignant"
        binary_conf = malignant_prob
    else:
        binary_label = "Benign"
        binary_conf = benign_prob

    binary_text = (
        f"Malignant vs Benign (EfficientNet-B3)\n"
        f"Prediction: {binary_label}  (probability = {binary_conf:.3f})\n"
        f"Malignant probability (MEL + BCC): {malignant_prob:.3f} | "
        f"Benign probability (NV, AKIEC, BKL, DF, VASC): {benign_prob:.3f}"
    )

    one_hot = [1 if i == top_idx else 0 for i in range(len(CLASS_LABELS))]
    multi_text = (
        "Seven-class lesion type (EfficientNet-B3)\n"
        f"Predicted class: {top_label}  (probability = {top_prob:.3f})\n"
        "Class index mapping: 0=MEL, 1=NV, 2=BCC, 3=AKIEC, 4=BKL, 5=DF, 6=VASC\n"
        f"One-hot encoding: {one_hot}"
    )

    # --------------------------------------------------------------
    # Grad-CAM with SCRATCH CNN (ONLY for interpretability)
    # --------------------------------------------------------------
    # Scratch CNN uses ISIC-specific normalization
    x_scratch = SCRATCH_TRANSFORM(img_resized).unsqueeze(0).to(DEVICE)

    # We let Grad-CAM use the scratch model's own predicted class
    cam = gradcam_scratch.generate_cam(x_scratch, class_idx=None)

    # Resize CAM to 224x224 and convert to heatmap
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR

    # Prepare original image for overlay in BGR
    orig_rgb = img_np
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(heatmap_color, 0.4, orig_bgr, 0.6, 0)

    # Convert back to RGB for Gradio display
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return (
        img_np,              # original (resized) image
        binary_text,         # malignant vs benign
        multi_text,          # 7-class + one-hot
        heatmap_rgb,         # Grad-CAM heatmap (scratch CNN)
        overlay_rgb,         # Overlay (scratch CNN)
        GRADCAM_EXPLANATION  # ~40-word explanation
    )


# ------------------------------------------------------------------
# 7. Gradio UI layout
# ------------------------------------------------------------------
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Skin Lesion Classification (ISIC-2018)

            - **Model B (EfficientNet-B3)** for prediction  
            - **Model A (Scratch CNN)** for Grad-CAM interpretability  
            """
        )

        with gr.Row():
            image_input = gr.Image(
                label="Upload dermoscopy image",
                type="filepath",
            )

        run_button = gr.Button("Run Analysis")

        with gr.Row():
            output_image = gr.Image(
                label="Input image (resized to 224×224)",
            )
            binary_box = gr.Textbox(
                label="Malignant vs Benign (EfficientNet-B3)",
                lines=4,
            )
            multi_box = gr.Textbox(
                label="Seven-class lesion prediction (EfficientNet-B3)",
                lines=5,
            )

        with gr.Row():
            heatmap_img = gr.Image(
                label="Grad-CAM heatmap (Scratch CNN)",
            )
            overlay_img = gr.Image(
                label="Grad-CAM overlay (Scratch CNN)",
            )

        gradcam_text_box = gr.Textbox(
            label="Importance of Grad-CAM for interpretability",
            value=GRADCAM_EXPLANATION,
            lines=3,
            interactive=False,
        )

        run_button.click(
            fn=analyze_image,
            inputs=image_input,
            outputs=[
                output_image,
                binary_box,
                multi_box,
                heatmap_img,
                overlay_img,
                gradcam_text_box,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="localhost", server_port=7860, share=False)
