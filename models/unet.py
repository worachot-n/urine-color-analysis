"""
U-Net for binary grid-line segmentation using segmentation_models_pytorch.

Architecture:
  Backbone : ResNet-34 (ImageNet pre-trained)
  Decoder  : U-Net decoder with skip connections
  Input    : 3-channel RGB image, any size (training: 640×640)
  Output   : 1-channel sigmoid mask  (grid lines ≈ 1, background ≈ 0)

Usage:
    from models.unet import build_model
    model = build_model(weights_path="models/unet_grid.pt")
    # model is in eval mode, outputs sigmoid probabilities
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(weights_path: str | None = None, device: str = "cpu") -> nn.Module:
    """
    Build U-Net with ResNet-34 backbone.

    Args:
        weights_path: Path to saved state-dict (.pt). If None, returns model with
                      ImageNet-pretrained encoder and randomly-initialised decoder.
        device:       Torch device string, e.g. "cpu" or "cuda".

    Returns:
        Model in eval() mode on the specified device.

    Raises:
        FileNotFoundError: if weights_path is given but the file does not exist.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )

    if weights_path is not None:
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"U-Net weights not found: {weights_path}\n"
                "Place the trained weights file at that path before starting the server."
            )
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# Convenience alias so existing imports of `UNet` still resolve
UNet = smp.Unet
