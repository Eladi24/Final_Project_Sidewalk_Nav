"""Sidewalk segmentation — Stage A (GPU only, runs on Colab, NOT on local MX330).

Supports two backends, selected automatically from the model name:
  SegFormer  — nvidia/segformer-b2-finetuned-cityscapes-1024-1024  (default)
  Mask2Former — facebook/mask2former-swin-tiny-mapillary-vistas-semantic

SegFormer architecture (Xie et al. 2021):
    Hierarchical transformer encoder with overlapping patch embeddings at
    multiple scales (4×, 8×, 16×, 32×), fused by a lightweight MLP decoder
    into a dense per-pixel class prediction.

Mask2Former architecture (Cheng et al. 2022):
    Universal segmentation transformer.  A pixel decoder produces multi-scale
    feature maps; a transformer decoder attends to them with learned object
    queries; post-processing converts query masks to a semantic map.

Stage contract:
    Input : RGB frame, uint8 HxWx3.
    Output: binary_mask, uint8 HxW, 255 = traversable, 0 = not traversable.
            Spatial size matches the input frame.

DO NOT import this module in Stage B code. Stage B reads cached .png files.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


class SidewalkSegmenter:
    """Wraps SegFormer or Mask2Former for per-frame sidewalk segmentation.

    The backend is auto-detected from the model name string:
        "mask2former" in model_name  →  Mask2Former pipeline
        anything else                →  SegFormer pipeline

    Class IDs are auto-discovered from the model's ``config.id2label`` by
    searching for "sidewalk", "road", and "pedestrian area" in the label
    strings.  This works for Cityscapes (19 classes), Mapillary Vistas
    (65 classes), and ADE20K (150 classes) without hardcoding anything.
    Pass ``traversable_class_ids`` explicitly to override.

    Args:
        model_name: HuggingFace model ID.
        traversable_class_ids: class indices to treat as walkable surface.
            ``None`` triggers auto-discovery (recommended).
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        traversable_class_ids: list[int] | None = None,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self._backend = "mask2former" if "mask2former" in model_name.lower() else "segformer"

        if self._backend == "mask2former":
            from transformers import (
                Mask2FormerForUniversalSegmentation,
                Mask2FormerImageProcessor,
            )
            self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        else:
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()

        # Build int-keyed id→label map (HuggingFace stores keys as strings)
        id2label: dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

        # Traversable classes: road + sidewalk + pedestrian area
        _traversable_keywords = ("road", "sidewalk", "pedestrian area", "bike lane",
                                  "parking", "service lane")
        if traversable_class_ids is not None:
            self.traversable_class_ids = traversable_class_ids
        else:
            self.traversable_class_ids = [
                k for k, v in id2label.items()
                if any(kw in v.lower() for kw in _traversable_keywords)
            ]
            print(f"[SidewalkSegmenter] Auto-discovered traversable classes:")
            for cid in sorted(self.traversable_class_ids):
                print(f"  {cid:3d}: {id2label[cid]}")

        # Sidewalk-only class: the narrower mask Stage B prefers for boundary
        # extraction.  Auto-detected as the first label containing "sidewalk".
        # run_inference_colab.py reads this attribute to save sidewalk_*.png.
        sidewalk_hits = [k for k, v in id2label.items() if "sidewalk" in v.lower()]
        self.sidewalk_only_class_id: int = sidewalk_hits[0] if sidewalk_hits else 1
        print(
            f"[SidewalkSegmenter] Sidewalk-only class: "
            f"{self.sidewalk_only_class_id} "
            f"({id2label.get(self.sidewalk_only_class_id, '?')})"
        )

    def segment(
        self,
        rgb_frame: np.ndarray,
        return_class_map: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Produce a binary traversable-surface mask for one RGB frame.

        Args:
            rgb_frame: uint8 array of shape (H, W, 3).
            return_class_map: if True, also return the raw per-pixel class map
                (int16, shape (H, W)) so callers can derive class-specific
                masks without running inference twice.

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
            outputs = self.model(**inputs)

        if self._backend == "mask2former":
            # post_process_semantic_segmentation handles upsampling internally
            # and returns a list of tensors (one per image in the batch).
            seg_map = (
                self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(H, W)]
                )[0]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
        else:
            # SegFormer outputs low-res logits; upsample manually then argmax.
            logits_up = torch.nn.functional.interpolate(
                outputs.logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            seg_map = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

        binary_mask = np.isin(seg_map, self.traversable_class_ids).astype(np.uint8) * 255

        if return_class_map:
            return binary_mask, seg_map.astype(np.int16)
        return binary_mask

    def print_class_labels(self) -> None:
        """Print the model's full class ID → label mapping.

        Run this once when switching models to verify which IDs were
        auto-discovered as traversable and which as sidewalk-only.
        """
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        print(f"\nModel class labels ({len(id2label)} total):")
        for k in sorted(id2label.keys()):
            tags = []
            if k in self.traversable_class_ids:
                tags.append("traversable")
            if k == self.sidewalk_only_class_id:
                tags.append("sidewalk-only")
            suffix = f"  ← {', '.join(tags)}" if tags else ""
            print(f"  {k:3d}: {id2label[k]}{suffix}")


if __name__ == "__main__":
    import cv2

    parser = argparse.ArgumentParser(description="Segment sidewalk in one image")
    parser.add_argument("image", help="Path to input RGB image (.png / .jpg)")
    parser.add_argument("--out", default="mask_preview.png",
                        help="Output mask overlay PNG (default: mask_preview.png)")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        help="HuggingFace model ID (SegFormer or Mask2Former)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--print-classes", action="store_true",
        help="Print the model's full class label map and exit",
    )
    args = parser.parse_args()

    segmenter = SidewalkSegmenter(model_name=args.model, device=args.device)

    if args.print_classes:
        segmenter.print_class_labels()
        sys.exit(0)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask, class_map = segmenter.segment(frame_rgb, return_class_map=True)
    sidewalk_mask = (class_map == segmenter.sidewalk_only_class_id).astype(np.uint8) * 255

    # Overlay: combined traversable = green tint, sidewalk-only = brighter green
    overlay = frame.copy()
    overlay[mask > 127] = (
        overlay[mask > 127] * 0.5 + np.array([0, 100, 0]) * 0.5
    ).astype(np.uint8)
    overlay[sidewalk_mask > 127] = (
        overlay[sidewalk_mask > 127] * 0.5 + np.array([0, 220, 0]) * 0.5
    ).astype(np.uint8)

    cv2.imwrite(args.out, overlay)
    print(f"Traversable coverage : {(mask > 0).mean() * 100:.1f}%")
    print(f"Sidewalk-only coverage: {(sidewalk_mask > 0).mean() * 100:.1f}%")
    print(f"Overlay saved to {args.out}")
