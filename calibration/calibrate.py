"""Zhang's checkerboard camera calibration.

Algorithm overview (Zhang 2000):
    1. Print a planar checkerboard and photograph it from ~20 different angles.
    2. Detect corners with sub-pixel accuracy (cv2.findChessboardCorners +
       cv2.cornerSubPix).
    3. cv2.calibrateCamera solves for the camera matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
       and distortion coefficients by minimising reprojection error via
       Levenberg-Marquardt over all views simultaneously.
    4. Each view contributes a homography H_i (3x3) relating the world checkerboard
       plane to the image.  Two linear constraints per view on K come from the
       orthogonality and unit-length conditions on the rotation columns derived
       from H_i = K [r1 r2 t].  With >= 3 views the system is over-determined and
       solved in closed form, then refined by non-linear optimisation.

Why intrinsics are resolution-dependent:
    fx, fy, cx, cy are all in pixels. If you scale the image by a factor s, you
    must also scale fx, fy, cx, cy by s. Always calibrate at the same resolution
    you will use for depth estimation.

Output: calibration/intrinsics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def calibrate(
    image_dir: str | Path,
    board_cols: int,
    board_rows: int,
    square_size_m: float,
    out_path: str | Path,
) -> float:
    """Run Zhang's calibration and write intrinsics to *out_path*.

    Args:
        image_dir: Directory containing checkerboard JPEG/PNG images.
        board_cols: Number of *inner* corners along the horizontal axis.
        board_rows: Number of *inner* corners along the vertical axis.
        square_size_m: Physical side length of one square in metres.
        out_path: Destination for the intrinsics JSON file.

    Returns:
        Mean reprojection error in pixels (lower is better; aim for < 1.0 px).
    """
    board_size = (board_cols, board_rows)

    # 3-D world coordinates for a single view (Z=0 on the checkerboard plane)
    objp = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []   # 3-D world points across all views
    img_points: list[np.ndarray] = []   # corresponding 2-D image points

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    image_size: tuple[int, int] | None = None
    found_count = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (W, H)

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            print(f"  [WARN] No checkerboard found in {img_path.name}, skipping.")
            continue

        # Refine to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners_refined)
        found_count += 1
        print(f"  [OK]   {img_path.name}")

    if found_count < 3:
        raise RuntimeError(
            f"Need at least 3 usable images; only found {found_count}. "
            "Ensure the checkerboard is fully visible and try more photos."
        )

    print(f"\nCalibrating with {found_count} views ...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": image_size[0],
        "height": image_size[1],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "reprojection_error_px": float(rms),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(intrinsics, fh, indent=2)

    print(f"Reprojection error: {rms:.4f} px")
    print(f"Intrinsics written to {out_path}")
    return rms


def load_intrinsics(path: str | Path) -> dict:
    """Load intrinsics JSON and return a plain dict with numpy arrays where needed."""
    with open(path, "r") as fh:
        data = json.load(fh)
    return {
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
        "width": int(data["width"]),
        "height": int(data["height"]),
        "camera_matrix": np.array(data["camera_matrix"], dtype=np.float64),
        "dist_coeffs": np.array(data["dist_coeffs"], dtype=np.float64),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zhang checkerboard calibration")
    parser.add_argument(
        "--images", default="calibration/images",
        help="Directory of checkerboard photos (default: calibration/images)"
    )
    parser.add_argument("--cols", type=int, default=9,
                        help="Inner corners along width (default: 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="Inner corners along height (default: 6)")
    parser.add_argument("--square", type=float, default=0.025,
                        help="Square size in metres (default: 0.025 = 2.5 cm)")
    parser.add_argument(
        "--out", default="calibration/intrinsics.json",
        help="Output JSON path (default: calibration/intrinsics.json)"
    )
    args = parser.parse_args()

    try:
        rms = calibrate(args.images, args.cols, args.rows, args.square, args.out)
        print(f"\nDone. RMS reprojection error = {rms:.4f} px")
        if rms > 1.0:
            print("WARNING: error > 1.0 px. Consider retaking photos with better coverage.")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
