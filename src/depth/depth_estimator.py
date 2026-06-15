"""Depth estimation — Stage A (GPU only, runs on Colab, NOT on local MX330).

Model: Depth Anything V2 — Metric Outdoor variant
    Paper: "Depth Anything V2" (Yang et al. 2024)
    HuggingFace: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf

Architecture overview:
    - Encoder: Vision Transformer (ViT-S/B) pretrained on DINOv2 features.
      The image is split into non-overlapping 14x14 patches; each patch is
      linearly embedded and processed with self-attention across the full patch
      sequence.
    - Decoder: DPT (Dense Prediction Transformer).  Reassemble + Fusion blocks
      progressively upsample and fuse multi-scale ViT feature maps back to
      the original resolution, producing a dense prediction head.

Relative vs metric depth:
    - *Relative* models (the default Depth Anything checkpoint) predict depth up
      to an unknown scale and shift:  d_pred = s * d_real + t.  The output
      values are dimensionless and cannot be used directly as metres.
    - *Metric* models are fine-tuned on datasets with known absolute depth
      (e.g. KITTI, NYUv2, outdoor-specific) and predict depth in metres
      without any scale factor.  We use the outdoor metric checkpoint so that
      obstacle distances can be compared to tape-measure ground truth.
    - Scale-invariant training: the relative model trains with a scale- and
      shift-invariant loss so that the network focuses purely on relative depth
      structure. The metric fine-tune then reattaches scale by training on
      real metric ground truth.

Stage contract:
    Input : RGB frame, uint8 HxWx3 (any resolution).
    Output: float32 HxW depth map in METRES, same spatial size as input.

DO NOT import this module in Stage B code. Stage B reads cached .npy files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class DepthEstimator:
    """Wraps Depth Anything V2 metric for per-frame depth inference.

    Args:
        model_name: HuggingFace model ID for the metric outdoor checkpoint.
        device: ``"cuda"`` or ``"cpu"`` (CPU is very slow; Colab T4 recommended).
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        device: str = "cuda",
    ) -> None:
        # Import torch here so Stage B code never triggers a GPU import
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def estimate(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Estimate per-pixel metric depth for one RGB frame.

        Args:
            rgb_frame: uint8 array of shape (H, W, 3), BGR or RGB — the
                processor handles normalisation internally.

        Returns:
            depth_m: float32 array of shape (H, W) with depth in metres.
                     The spatial dimensions match the input frame exactly
                     (the model's internal resolution is handled transparently).
        """
        import torch
        from PIL import Image

        H, W = rgb_frame.shape[:2]

        # Convert to PIL (processor expects PIL or HF image input)
        pil_img = Image.fromarray(rgb_frame)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # predicted_depth: (1, H_model, W_model) — upsampled to original size
        pred = outputs.predicted_depth  # shape (1, H', W')

        # Upsample back to original frame resolution
        pred_up = torch.nn.functional.interpolate(
            pred.unsqueeze(1),           # (1, 1, H', W')
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze()                      # (H, W)

        depth_m = pred_up.cpu().numpy().astype(np.float32)
        return depth_m


if __name__ == "__main__":
    import sys
    import cv2

    parser = argparse.ArgumentParser(description="Estimate depth for one image")
    parser.add_argument("image", help="Path to input RGB image")
    parser.add_argument("--out", default="depth_preview.png",
                        help="Output colorized depth PNG (default: depth_preview.png)")
    parser.add_argument("--model",
                        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-npy", action="store_true",
                        help="Also save raw float32 depth as .npy alongside --out")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    estimator = DepthEstimator(model_name=args.model, device=args.device)
    depth_m = estimator.estimate(frame_rgb)

    print(f"Depth range: {depth_m.min():.2f} m — {depth_m.max():.2f} m")

    # Colorize for inspection
    depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    cv2.imwrite(args.out, colored)
    print(f"Colorized depth saved to {args.out}")

    if args.save_npy:
        npy_path = Path(args.out).with_suffix(".npy")
        np.save(npy_path, depth_m)
        print(f"Raw depth saved to {npy_path}")
