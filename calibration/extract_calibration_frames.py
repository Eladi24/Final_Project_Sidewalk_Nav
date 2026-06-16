"""Extract a diverse, sharp set of calibration frames from a video.

Why not just grab every Nth frame:
    Zhang's calibration needs views from genuinely different poses (distance,
    tilt, rotation) to disambiguate the intrinsics — if every selected frame
    shows nearly the same pose, the system is under-determined and the
    solution is unstable. Naive uniform sampling of a video tends to include
    near-duplicate, blurry, or out-of-frame shots, especially around the
    start/end of a hand-held pan.

Approach:
    1. Walk every frame (or every --step-th frame on long videos) and attempt
       pattern detection right away (cv2.findChessboardCorners or
       findCirclesGrid, same code path as calibrate.py). This is itself a
       strong filter: blurry or off-board frames simply fail detection and
       are discarded, so every candidate is already known to be usable.
    2. Score each successful candidate by sharpness (Laplacian variance —
       a focused image has high-frequency edges that Laplacian responds to
       strongly; blur smooths those edges and the variance drops).
    3. Bucket candidates evenly across the video's frame range and keep the
       sharpest candidate per bucket. This spreads the final selection across
       the whole pan instead of clustering wherever detection happens to be
       easiest.

Usage:
    python calibration/extract_calibration_frames.py \\
        --video calibration/raw/checkerboard.mp4 \\
        --out calibration/images \\
        --cols 7 --rows 5 --pattern chessboard \\
        --num-frames 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import _detect_pattern  # reuse the exact detection code calibrate.py uses


def _sharpness(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness score (higher = sharper / more in-focus)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    board_cols: int,
    board_rows: int,
    pattern: str = "chessboard",
    num_frames: int = 30,
    step: int = 1,
) -> int:
    """Scan a calibration video and write the best ~num_frames frames to disk.

    Args:
        video_path: Input video file.
        out_dir: Output directory for selected frame images (created if needed).
        board_cols: Inner corners (chessboard) or dots (circles) along width.
        board_rows: Same, along height.
        pattern: ``"chessboard"`` or ``"circles"``.
        num_frames: Target number of frames to select.
        step: Process every step-th frame while scanning (1 = every frame;
              raise this on very long videos to speed up scanning).

    Returns:
        Number of frames actually written (may be less than num_frames if
        the pattern wasn't detected in enough distinct poses).
    """
    board_size = (board_cols, board_rows)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Scanning {total_frames} frames (step={step}) for pattern detections ...")

    # candidates: list of (frame_index, sharpness, bgr_frame)
    candidates: list[tuple[int, float, np.ndarray]] = []

    frame_idx = 0
    scanned = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, _ = _detect_pattern(gray, board_size, pattern)
            if found:
                score = _sharpness(gray)
                candidates.append((frame_idx, score, frame.copy()))
            scanned += 1
            if scanned % 100 == 0:
                print(f"  scanned {scanned} frames, {len(candidates)} detections so far ...")

        frame_idx += 1

    cap.release()

    if not candidates:
        raise RuntimeError(
            "No frames with a detected pattern were found. Check --cols/--rows/--pattern "
            "match your actual board, and that the board is in frame and reasonably sharp."
        )

    print(f"\n{len(candidates)} / {scanned} scanned frames had a valid detection.")

    # Bucket candidates evenly across the frame index range, keep sharpest per bucket
    first_idx = candidates[0][0]
    last_idx = candidates[-1][0]
    span = max(last_idx - first_idx, 1)
    n_buckets = min(num_frames, len(candidates))

    buckets: dict[int, tuple[int, float, np.ndarray]] = {}
    for idx, score, frame in candidates:
        bucket = min(int((idx - first_idx) / span * n_buckets), n_buckets - 1)
        if bucket not in buckets or score > buckets[bucket][1]:
            buckets[bucket] = (idx, score, frame)

    selected = sorted(buckets.values(), key=lambda t: t[0])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (idx, score, frame) in enumerate(selected):
        out_path = out_dir / f"calib_{i:03d}.png"
        cv2.imwrite(str(out_path), frame)

    print(f"Selected {len(selected)} frames (target was {num_frames}).")
    print(f"Saved to {out_dir}")
    if len(selected) < num_frames:
        print(
            f"WARNING: only found {len(selected)} usable frames. Calibration needs >= 3, "
            "but more (15-30) spread across angles gives a much more stable result. "
            "Consider filming a longer/wider pan."
        )
    return len(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract diverse, sharp calibration frames from a video"
    )
    parser.add_argument("--video", required=True, help="Input calibration video")
    parser.add_argument(
        "--out", default="calibration/images",
        help="Output directory for selected frames (default: calibration/images)"
    )
    parser.add_argument("--cols", type=int, default=7,
                        help="Inner corners (chessboard) or dots (circles) along width")
    parser.add_argument("--rows", type=int, default=5,
                        help="Inner corners (chessboard) or dots (circles) along height")
    parser.add_argument(
        "--pattern", choices=["chessboard", "circles"], default="chessboard",
        help="'chessboard' (default, recommended) or 'circles' for a dot grid"
    )
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Target number of frames to select (default: 30)")
    parser.add_argument("--step", type=int, default=1,
                        help="Scan every step-th frame (default: 1 = every frame; "
                             "increase for very long videos to speed up scanning)")
    args = parser.parse_args()

    try:
        extract_frames(
            args.video, args.out, args.cols, args.rows,
            pattern=args.pattern, num_frames=args.num_frames, step=args.step,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
