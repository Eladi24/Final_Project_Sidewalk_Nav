"""Stage B demo entry point — runs entirely on CPU, no GPU required.

Usage:
    python scripts/run_video.py \\
        --cache data/cache/sample \\
        --config configs/default.yaml \\
        --out output/sample.mp4

The script:
    1. Reads the cached frame/depth/mask triplets from data/cache/<clip>/.
    2. Feeds each triplet to Pipeline.process_frame in order.
    3. Writes the annotated video to --out with cv2.VideoWriter.
    4. Optionally streams spoken alerts via AudioAlerter if --audio is set.

Cache layout expected (written by Stage A / run_inference_colab.py):
    <cache>/frame_00000.png
    <cache>/depth_00000.npy    (float32 HxW, metres)
    <cache>/mask_00000.png     (uint8 HxW, 255=sidewalk)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gc

import cv2
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import Pipeline


def collect_frame_indices(cache_dir: Path) -> list[int]:
    """Return sorted list of integer indices for which all three files exist."""
    indices = []
    for f in sorted(cache_dir.glob("frame_*.png")):
        idx = int(f.stem.split("_")[1])
        depth_path = cache_dir / f"depth_{idx:05d}.npy"
        mask_path = cache_dir / f"mask_{idx:05d}.png"
        if depth_path.exists() and mask_path.exists():
            indices.append(idx)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage B pipeline on a cached clip and produce an annotated video"
    )
    parser.add_argument(
        "--cache", required=True,
        help="Cache directory produced by run_inference_colab.py (e.g. data/cache/sample)"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Config YAML path (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--out", default="output/annotated.mp4",
        help="Output video path (default: output/annotated.mp4)"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Output video frame rate (default: 30)"
    )
    parser.add_argument(
        "--audio", action="store_true",
        help="Enable spoken distance alerts via pyttsx3"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames (useful for quick tests)"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    if not cache_dir.is_dir():
        print(f"ERROR: cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    pipeline = Pipeline(cfg)

    indices = collect_frame_indices(cache_dir)
    if not indices:
        print(f"ERROR: no complete frame/depth/mask triplets in {cache_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_frames:
        indices = indices[: args.max_frames]

    print(f"Processing {len(indices)} frames from {cache_dir} ...")

    # Initialise VideoWriter using the first frame's dimensions
    first_frame = cv2.imread(str(cache_dir / f"frame_{indices[0]:05d}.png"))
    if first_frame is None:
        print("ERROR: cannot read first frame.", file=sys.stderr)
        sys.exit(1)
    H, W = first_frame.shape[:2]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))

    alerter = None
    if args.audio:
        from src.output.audio_alerts import AudioAlerter
        alerter = AudioAlerter()

    for frame_no, idx in enumerate(indices):
        frame = cv2.imread(str(cache_dir / f"frame_{idx:05d}.png"))
        depth = np.load(cache_dir / f"depth_{idx:05d}.npy")
        mask = cv2.imread(str(cache_dir / f"mask_{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)

        if frame is None or mask is None:
            print(f"  [WARN] Skipping frame {idx} (missing file).")
            continue

        annotated, tracks = pipeline.process_frame(frame, depth, mask, frame_index=idx)

        # If the pipeline ran at reduced resolution, upscale back for the video
        if annotated.shape[:2] != (H, W):
            annotated = cv2.resize(annotated, (W, H), interpolation=cv2.INTER_LINEAR)

        writer.write(annotated)

        if alerter and tracks:
            phrase = alerter.maybe_speak(tracks)
            if phrase:
                print(f"  frame {idx:05d}: {phrase}")

        if (frame_no + 1) % 50 == 0:
            print(f"  {frame_no + 1}/{len(indices)} frames done …")
            gc.collect()

    writer.release()
    print(f"Annotated video written to {out_path}")


if __name__ == "__main__":
    main()
