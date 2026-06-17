"""Sidewalk segmentation — Stage A (GPU only, runs on Colab, NOT on local MX330).

Model: SegFormer-B2 fine-tuned on Cityscapes (150 classes → 19 classes)
    HuggingFace: nvidia/segformer-b2-finetuned-cityscapes-1024-1024

Architecture:
    SegFormer (Xie et al. 2021) is a hierarchical transformer encoder with a
    lightweight MLP decoder.  Unlike ViT, it uses overlapping patch embeddings
    at multiple scales (4x, 8x, 16x, 32x downsampling), producing multi-scale
    feature maps that the decoder fuses into a single dense prediction.

Cityscapes label 1 = 'sidewalk':
    Classes: road(0), sidewalk(1), building(2), wall(3), fence(4), pole(5),
    traffic light(6), traffic sign(7), vegetation(8), terrain(9), sky(10),
    person(11), rider(12), car(13), truck(14), bus(15), train(16),
    motorcycle(17), bicycle(18).

Stage contract:
    Input : RGB frame, uint8 HxWx3.
    Output: binary_mask, uint8 HxW, 255 = sidewalk, 0 = not sidewalk.
            Spatial size matches the input frame.

DO NOT import this module in Stage B code. Stage B reads cached .png files.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


class SidewalkSegmenter:
    """Wraps SegFormer-B2 Cityscapes for per-frame sidewalk segmentation.

    Args:
        model_name: HuggingFace model ID.
        traversable_class_ids: list of Cityscapes class indices to treat as
            walkable surface.  Default ``[1]`` (sidewalk only).  For footage
            where the pavement is classified as road (class 0) — common with
            cobblestone / brick sidewalks that differ from Cityscapes training
            data — pass ``[0, 1]`` to include both road and sidewalk.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        traversable_class_ids: list[int] | None = None,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = device
        self.traversable_class_ids = traversable_class_ids if traversable_class_ids is not None else [1]
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def segment(
        self,
        rgb_frame: np.ndarray,
        return_class_map: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Produce a binary traversable-surface mask for one RGB frame.

        Args:
            rgb_frame: uint8 array of shape (H, W, 3).
            return_class_map: if True, also return the raw per-pixel class map
                (int16, shape (H, W)) so callers can derive class-specific masks
                (e.g. class 1 = true sidewalk) without running inference twice.

        Returns:
            binary_mask: uint8 (H, W), 255 where any traversable class is predicted.
            class_map (only when return_class_map=True): int16 (H, W), raw class index.
        """
        import torch
        from PIL import Image

        H, W = rgb_frame.shape[:2]
        pil_img = Image.fromarray(rgb_frame)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits   # (1, num_classes, H/4, W/4)

        logits_up = torch.nn.functional.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, num_classes, H, W)

        seg_map = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W) int64

        binary_mask = np.isin(seg_map, self.traversable_class_ids).astype(np.uint8) * 255
        if return_class_map:
            return binary_mask, seg_map.astype(np.int16)
        return binary_mask


if __name__ == "__main__":
    import cv2

    parser = argparse.ArgumentParser(description="Segment sidewalk in one image")
    parser.add_argument("image", help="Path to input RGB image")
    parser.add_argument("--out", default="mask_preview.png",
                        help="Output mask overlay PNG (default: mask_preview.png)")
    parser.add_argument("--model",
                        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    parser.add_argument("--sidewalk-class", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    segmenter = SidewalkSegmenter(
        model_name=args.model,
        sidewalk_class_id=args.sidewalk_class,
        device=args.device,
    )
    mask = segmenter.segment(frame_rgb)

    # Overlay: tint sidewalk pixels green
    overlay = frame.copy()
    overlay[mask > 127] = (overlay[mask > 127] * 0.5 + np.array([0, 128, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(args.out, overlay)
    sidewalk_frac = (mask > 0).mean() * 100
    print(f"Sidewalk coverage: {sidewalk_frac:.1f}%")
    print(f"Overlay saved to {args.out}")
