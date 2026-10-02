"""
Export EfficientNet-B0 Diabetic Retinopathy Checkpoint to ONNX for Mobile Edge AI.

This script exports the verified EfficientNet-B0 checkpoint (67d0b896...) to an
optimized ONNX model that runs purely on-device (client-side) using onnxruntime-web
or native mobile runtimes (Capacitor / Android / iOS).

Outputs:
  - logits: (batch_size, 5) ICDR classification raw outputs (Softmax applied in JS)
  - features: (batch_size, 1280, 7, 7) final convolutional feature maps (features.8)
    for real-time on-device Grad-CAM generation without any server connection.
  - classifier_weights.json: weight matrix of the Linear(1280, 5) head to allow
    instant client-side Class Activation Map weighting.
"""

import json
import os
import sys
import hashlib
import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "weights", "efficientnet_b0_dr.pth")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "frontend", "public", "models")
EXPECTED_SHA256 = "67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7"


class EfficientNetB0EdgeWrapper(nn.Module):
    """
    Dual-output wrapper for EfficientNet-B0:
    Returns both classification logits and the final spatial feature map
    for zero-dependency client-side Grad-CAM rendering.
    """

    def __init__(self, base_model: models.EfficientNet):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

    def forward(self, x: torch.Tensor):
        # Forward through feature extractor
        feature_maps = self.features(x)  # (B, 1280, 7, 7)
        pooled = self.avgpool(feature_maps)  # (B, 1280, 1, 1)
        flattened = torch.flatten(pooled, 1)  # (B, 1280)
        logits = self.classifier(flattened)  # (B, 5)
        return logits, feature_maps


def export_to_onnx(output_dir: str = DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "efficientnet_b0_dr.onnx")
    weights_json_path = os.path.join(output_dir, "classifier_weights.json")

    # 1. Verify Checkpoint Digest
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    digest = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    actual_hash = digest.hexdigest().lower()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"Checkpoint digest mismatch.\nExpected: {EXPECTED_SHA256}\nActual:   {actual_hash}"
        )
    print(f"Verified checkpoint SHA-256: {actual_hash}")

    # 2. Reconstruct Model Topology
    base_model = models.efficientnet_b0(weights=None)
    in_features = base_model.classifier[1].in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=in_features, out_features=5, bias=True),
    )

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    clean_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    base_model.load_state_dict(clean_dict, strict=True)
    base_model.eval()

    # Extract classifier linear layer weights for CAM
    linear_layer: nn.Linear = base_model.classifier[1]
    classifier_weights = {
        "weight": linear_layer.weight.detach().cpu().numpy().tolist(),  # (5, 1280)
        "bias": linear_layer.bias.detach().cpu().numpy().tolist(),      # (5,)
        "classes": [
            "0 - No DR",
            "1 - Mild NPDR",
            "2 - Moderate NPDR",
            "3 - Severe NPDR",
            "4 - PDR"
        ]
    }
    with open(weights_json_path, "w") as f:
        json.dump(classifier_weights, f)
    print(f"Saved classifier weights to: {weights_json_path}")

    # 3. Wrap for Edge Export
    wrapped_model = EfficientNetB0EdgeWrapper(base_model)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # 4. Perform ONNX Export
    print(f"Exporting to ONNX: {onnx_path} ...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
    )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"SUCCESS: ONNX edge model exported ({file_size_mb:.2f} MB).")
    print(f"Location: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    export_to_onnx(out_dir)
