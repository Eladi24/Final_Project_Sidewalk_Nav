"""Quantitative evaluation against tape-measure ground truth — Stage B.

Compares the pipeline's predicted obstacle distance against real distances
measured with a tape measure in the field.  Reports:
    - Per-frame predicted vs measured distance.
    - Mean absolute error (MAE) and median absolute error.
    - Per-camera scale correction factor (useful for correcting monocular
      depth scale drift — see "Known risks" in Claude.md).

Ground-truth file format (JSON):
    A list of objects, one per labeled frame:
    [
      {"frame_index": 42, "cache_dir": "data/cache/clip1", "distance_m": 2.35},
      {"frame_index": 87, "cache_dir": "data/cache/clip1", "distance_m": 1.10},
      ...
    ]

Scale-correction factor:
    Monocular metric depth can carry a systematic scale error of 10-20%.
    We estimate it as:
        scale = median(measured_i / predicted_i)
    Multiplying predicted distances by `scale` reduces the systematic bias.
    Surface this honestly in the presentation rather than silently applying it.

Usage:
    python scripts/evaluate.py \\
        --gt data/ground_truth.json \\
        --config configs/default.yaml \\
        --intrinsics calibration/intrinsics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import Pipeline


def evaluate(
    ground_truth: list[dict],
    config: dict,
    intrinsics_path: str,
) -> dict:
    """Run the pipeline on labeled frames and compare to tape measurements.

    Args:
        ground_truth: list of dicts with keys ``frame_index``, ``cache_dir``,
                      ``distance_m`` (the tape-measured real distance).
        config: nested config dict from load_config.
        intrinsics_path: path to intrinsics JSON (used in config override).

    Returns:
        Results dict with keys ``rows``, ``mae``, ``median_ae``, ``scale_factor``.
    """
    config["camera"]["intrinsics"] = intrinsics_path
    pipeline = Pipeline(config)

    rows: list[dict] = []

    for entry in ground_truth:
        idx = int(entry["frame_index"])
        cache = Path(entry["cache_dir"])
        measured = float(entry["distance_m"])

        depth_path = cache / f"depth_{idx:05d}.npy"
        mask_path = cache / f"mask_{idx:05d}.png"
        frame_path = cache / f"frame_{idx:05d}.png"

        if not (depth_path.exists() and mask_path.exists() and frame_path.exists()):
            print(f"  [SKIP] frame {idx} in {cache} — missing files")
            continue

        frame = cv2.imread(str(frame_path))
        depth = np.load(depth_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        _, tracks = pipeline.process_frame(frame, depth, mask, frame_index=idx)

        if not tracks:
            print(f"  [WARN] frame {idx}: no obstacles detected (measured={measured:.2f}m)")
            rows.append({
                "frame_index": idx,
                "cache_dir": str(cache),
                "measured_m": measured,
                "predicted_m": None,
                "error_m": None,
            })
            continue

        # Use the nearest tracked obstacle as the predicted distance
        predicted = tracks[0].distance_m
        error = abs(predicted - measured)

        rows.append({
            "frame_index": idx,
            "cache_dir": str(cache),
            "measured_m": measured,
            "predicted_m": round(predicted, 3),
            "error_m": round(error, 3),
        })
        print(
            f"  frame {idx:05d}: measured={measured:.2f}m  "
            f"predicted={predicted:.2f}m  error={error:.2f}m"
        )

    # Filter rows with valid predictions
    valid = [r for r in rows if r["predicted_m"] is not None]
    if not valid:
        return {"rows": rows, "mae": None, "median_ae": None, "scale_factor": None}

    errors = np.array([r["error_m"] for r in valid])
    measured_arr = np.array([r["measured_m"] for r in valid])
    predicted_arr = np.array([r["predicted_m"] for r in valid])

    mae = float(errors.mean())
    median_ae = float(np.median(errors))
    # Scale factor: how much to multiply predictions to match measurements
    scale_factor = float(np.median(measured_arr / predicted_arr))

    return {
        "rows": rows,
        "mae": round(mae, 3),
        "median_ae": round(median_ae, 3),
        "scale_factor": round(scale_factor, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline distance accuracy against tape-measure ground truth"
    )
    parser.add_argument(
        "--gt", required=True,
        help="JSON file with ground-truth distance labels"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--intrinsics", default="calibration/intrinsics.json")
    parser.add_argument("--out", default=None,
                        help="Optional JSON path to save results (default: print only)")
    args = parser.parse_args()

    with open(args.gt) as fh:
        gt = json.load(fh)

    cfg = load_config(args.config)
    results = evaluate(gt, cfg, args.intrinsics)

    print("\n--- Evaluation Summary ---")
    print(f"  Frames evaluated : {sum(1 for r in results['rows'] if r['predicted_m'] is not None)}"
          f" / {len(results['rows'])}")
    if results["mae"] is not None:
        print(f"  MAE              : {results['mae']:.3f} m")
        print(f"  Median AE        : {results['median_ae']:.3f} m")
        print(f"  Scale factor     : {results['scale_factor']:.4f}  "
              f"(multiply predictions by this to reduce systematic bias)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
