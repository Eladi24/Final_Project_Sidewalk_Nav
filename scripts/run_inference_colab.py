"""Stage A — neural inference runner (GPU / Google Colab only).

Run this script on Google Colab (T4 GPU) to convert a raw video into the
per-frame cache that Stage B consumes.

Usage on Colab:
    # Mount Google Drive first
    from google.colab import drive
    drive.mount('/content/drive')

    # Then run:
    !python scripts/run_inference_colab.py \\
        --video /content/drive/MyDrive/clips/sidewalk.mp4 \\
        --out   /content/drive/MyDrive/cache/sidewalk \\
        --every 1

    # Or paste the script body directly into a Colab cell.

Output layout (per Claude.md data contract):
    <out>/frame_00000.png      (original RGB frame)
    <out>/depth_00000.npy      (float32 HxW, metric depth in metres)
    <out>/mask_00000.png       (uint8 HxW, 255=sidewalk)

Important:
    - Colab's local disk is wiped on session reset — always write to Drive.
    - Install deps first:  pip install -r requirements-colab.txt
    - Do NOT run this file locally on the MX330 (2 GB VRAM is insufficient).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def run_inference(
    video_path: str | Path,
    out_dir: str | Path,
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    seg_model: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    traversable_class_ids: list[int] | None = None,
    device: str = "cuda",
    every_nth: int = 1,
) -> None:
    """Extract frames from *video_path*, run depth + segmentation, write cache.

    Args:
        video_path: Input video file (.mp4, .mov, etc.).
        out_dir: Output directory for the cache (created if needed).
        depth_model: HuggingFace model ID for Depth Anything V2 metric.
        seg_model: HuggingFace model ID for SegFormer Cityscapes.
        traversable_class_ids: Cityscapes class indices to treat as walkable.
            Default ``[0, 1]`` (road + sidewalk). Cobblestone/brick sidewalks
            are typically classified as road (0) by a Cityscapes-trained model.
        device: ``"cuda"`` (recommended) or ``"cpu"``.
        every_nth: process only every N-th frame (1 = every frame).
    """
    # Late imports — keep torch out of Stage B code paths
    from src.depth.depth_estimator import DepthEstimator
    from src.segmentation.sidewalk_seg import SidewalkSegmenter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading models on {device} ...")
    depth_est = DepthEstimator(model_name=depth_model, device=device)
    # Pass traversable_class_ids=None to trigger auto-discovery from the model's
    # id2label config. This is required for non-Cityscapes models (e.g. Mapillary)
    # where class 0 and 1 are not road and sidewalk.
    seg = SidewalkSegmenter(
        model_name=seg_model,
        traversable_class_ids=traversable_class_ids,  # None = auto-discover
        device=device,
    )
    print(f"Traversable classes: {seg.traversable_class_ids}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")

    written = 0
    raw_idx = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        if raw_idx % every_nth != 0:
            raw_idx += 1
            continue

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frame_idx = written

        # --- Depth estimation ---
        depth_m = depth_est.estimate(rgb_frame)   # float32 (H, W)

        # --- Sidewalk segmentation ---
        # return_class_map=True gives us the raw class indices alongside the
        # combined binary mask, so we can also save a sidewalk-only mask (class 1)
        # without running the model twice.
        mask, class_map = seg.segment(rgb_frame, return_class_map=True)
        # sidewalk_only_class_id is auto-discovered from the model's id2label
        # config (class 1 for Cityscapes, auto-detected for Mapillary / others).
        # Stage B uses this narrower mask for corridor boundary extraction so the
        # walkable corridor doesn't bleed into the car road.
        sidewalk_mask = (class_map == seg.sidewalk_only_class_id).astype(np.uint8) * 255

        # --- Write outputs ---
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.png"), bgr_frame)
        np.save(out_dir / f"depth_{frame_idx:05d}.npy", depth_m.astype(np.float16))
        cv2.imwrite(str(out_dir / f"mask_{frame_idx:05d}.png"), mask)
        cv2.imwrite(str(out_dir / f"sidewalk_{frame_idx:05d}.png"), sidewalk_mask)

        written += 1
        raw_idx += 1

        if written % 20 == 0:
            print(f"  {written} frames written (raw frame {raw_idx}/{total_frames}) ...")

    cap.release()
    print(f"Done. {written} frames written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage A: run depth + segmentation on a video and write cache"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", required=True,
                        help="Output cache directory (e.g. data/cache/clip_name)")
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    )
    parser.add_argument(
        "--seg-model",
        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    )
    parser.add_argument(
        "--traversable-classes", type=int, nargs="+", default=None,
        help="Class IDs to treat as walkable. Omit to auto-discover from the model's "
             "label config (recommended for non-Cityscapes models like Mapillary). "
             "For SegFormer Cityscapes pass: --traversable-classes 0 1"
    )
    parser.add_argument("--device", default="cuda",
                        help="'cuda' (recommended on Colab) or 'cpu'")
    parser.add_argument(
        "--every", type=int, default=1,
        help="Process every N-th frame (default 1 = all frames)"
    )
    args = parser.parse_args()

    # Add repo root to path so src.* imports work when run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    try:
        run_inference(
            video_path=args.video,
            out_dir=args.out,
            depth_model=args.depth_model,
            seg_model=args.seg_model,
            traversable_class_ids=args.traversable_classes,
            device=args.device,
            every_nth=args.every,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
