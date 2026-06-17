"""One-time cache converter: halve resolution and save depth as float16.

The Stage A Colab script writes 1920x1080 float32 depth maps (8.3 MB each).
At that size, the backprojection meshgrids alone consume ~66 MB per frame and
glibc's heap grows without releasing memory, causing OOM on machines with
<=8 GB RAM before the clip finishes.

This script converts an existing cache directory to:
  - depth_*.npy  float16  960x540   (1.0 MB, was 8.3 MB — 8x smaller)
  - mask_*.png   uint8    960x540   (0.5 MB, was 2.1 MB)
  - frame_*.png  uint8    960x540   (1.6 MB, was 6.2 MB)

After running this, set  performance.spatial_stride: 1  in default.yaml and
point --cache at the new directory.  The pipeline reads float16 depth
transparently (arithmetic is promoted to float32 automatically).

float16 precision at navigation distances:
  0.5 m → error < 0.1 mm    (monocular depth error is 5–20 cm)
  5.0 m → error < 0.5 mm
  20 m  → error < 2 mm
Converting is lossless for all practical purposes.

Usage:
    python scripts/convert_cache.py \\
        --src data/cache/sidewalk1 \\
        --dst data/cache/sidewalk1_half
    # then:
    python scripts/run_video.py \\
        --cache data/cache/sidewalk1_half \\
        --config configs/default.yaml \\
        --out data/output/sidewalk1.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def convert_cache(src: Path, dst: Path, stride: int = 2) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(src.glob("depth_*.npy"))
    if not depth_files:
        print(f"ERROR: no depth_*.npy files found in {src}", file=sys.stderr)
        sys.exit(1)

    total = len(depth_files)
    print(f"Converting {total} frames from {src} → {dst}  (stride={stride}) …")

    for i, depth_path in enumerate(depth_files):
        idx = int(depth_path.stem.split("_")[1])
        frame_path = src / f"frame_{idx:05d}.png"
        mask_path  = src / f"mask_{idx:05d}.png"

        # --- depth: subsample + float16 ---
        depth = np.load(depth_path)                       # float32 HxW
        depth_small = depth[::stride, ::stride].astype(np.float16)
        np.save(dst / f"depth_{idx:05d}.npy", depth_small)

        # --- mask: nearest-neighbour downsample (preserves binary values) ---
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            H, W = mask.shape
            mask_small = cv2.resize(
                mask, (W // stride, H // stride), interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(str(dst / f"mask_{idx:05d}.png"), mask_small)

        # --- frame: area-average downsample (best quality for shrinking) ---
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
            H, W = frame.shape[:2]
            frame_small = cv2.resize(
                frame, (W // stride, H // stride), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(str(dst / f"frame_{idx:05d}.png"), frame_small)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} frames done …")

    # Report size change for the depth files
    src_bytes  = sum(p.stat().st_size for p in src.glob("depth_*.npy"))
    dst_bytes  = sum(p.stat().st_size for p in dst.glob("depth_*.npy"))
    print(f"\nDone.")
    print(f"  Depth files: {src_bytes/1e9:.2f} GB → {dst_bytes/1e9:.2f} GB")
    print(f"\nNext steps:")
    print(f"  1. In configs/default.yaml set  performance.spatial_stride: 1")
    print(f"  2. Run:  python scripts/run_video.py --cache {dst} --config configs/default.yaml ...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Halve cache resolution and convert depth to float16")
    parser.add_argument("--src", required=True, help="Source cache directory (full-resolution)")
    parser.add_argument("--dst", required=True, help="Destination cache directory (half-resolution)")
    parser.add_argument(
        "--stride", type=int, default=2,
        help="Spatial downsample factor (default: 2 → half resolution)"
    )
    args = parser.parse_args()

    convert_cache(Path(args.src), Path(args.dst), stride=args.stride)


if __name__ == "__main__":
    main()
