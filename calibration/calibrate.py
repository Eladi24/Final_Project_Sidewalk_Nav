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


def _detect_pattern(
    gray: np.ndarray,
    board_size: tuple[int, int],
    pattern: str,
) -> tuple[bool, np.ndarray | None]:
    """Detect calibration pattern points in a grayscale image.

    Args:
        gray: uint8 grayscale image.
        board_size: (cols, rows) of inner corners (chessboard) or dots (circles).
        pattern: ``"chessboard"`` or ``"circles"``.

    Returns:
        (found, points) — points is (N, 1, 2) float32 pixel coords, or None.
    """
    if pattern == "chessboard":
        return cv2.findChessboardCorners(gray, board_size, None)

    if pattern == "circles":
        # Hand-drawn / printed dot grids are rarely uniform circles, so use a
        # permissive blob detector rather than OpenCV's strict defaults.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 50_000
        params.filterByCircularity = True
        params.minCircularity = 0.5   # permissive — hand-drawn dots aren't perfect circles
        params.filterByConvexity = True
        params.minConvexity = 0.7
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        return cv2.findCirclesGrid(
            gray, board_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector
        )

    raise ValueError(f"Unknown pattern type: {pattern!r} (expected 'chessboard' or 'circles')")


def _save_coverage_map(
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    out_path: str | Path,
) -> None:
    """Plot every detected pattern point from every image on one blank canvas.

    Good coverage means points are spread across the WHOLE frame, including
    the edges and corners — lens distortion (the dist_coeffs) is strongest at
    the periphery, so if every detection clusters near the image centre, the
    distortion estimate for the edges is poorly constrained even if the
    overall RMS error looks fine.
    """
    canvas = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)
    for pts in img_points:
        for p in pts.reshape(-1, 2):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 4, (0, 0, 220), -1)
    cv2.imwrite(str(out_path), canvas)


def calibrate(
    image_dir: str | Path,
    board_cols: int,
    board_rows: int,
    square_size_m: float,
    out_path: str | Path,
    pattern: str = "chessboard",
    debug_dir: str | Path | None = None,
) -> float:
    """Run Zhang's calibration and write intrinsics to *out_path*.

    Args:
        image_dir: Directory containing checkerboard JPEG/PNG images.
        board_cols: Number of *inner* corners (chessboard) or dots (circles)
                    along the horizontal axis.
        board_rows: Same, along the vertical axis.
        square_size_m: Physical spacing between adjacent points in metres
                       (square side for chessboard, dot-to-dot spacing for
                       circles). Measure this directly — it scales every
                       distance the whole pipeline ever outputs.
        out_path: Destination for the intrinsics JSON file.
        pattern: ``"chessboard"`` (default) or ``"circles"`` for a dot grid.
        debug_dir: if given, write diagnostic visuals here:
                   - ``corners_<name>.png`` per accepted image, with detected
                     points drawn and connected (cv2.drawChessboardCorners) —
                     confirms the detector found the right points in the
                     right grid order.
                   - ``coverage_map.png`` — every detected point from every
                     image pooled onto one canvas, to check spatial coverage.

    Returns:
        Mean reprojection error in pixels (lower is better; aim for < 1.0 px).
    """
    board_size = (board_cols, board_rows)

    # 3-D world coordinates for a single view (Z=0 on the calibration plane)
    objp = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []   # 3-D world points across all views
    img_points: list[np.ndarray] = []   # corresponding 2-D image points
    image_names: list[str] = []         # filenames, parallel to obj/img_points

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    debug_dir_path = Path(debug_dir) if debug_dir else None
    if debug_dir_path:
        debug_dir_path.mkdir(parents=True, exist_ok=True)

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

        ret, corners = _detect_pattern(gray, board_size, pattern)
        if not ret:
            print(f"  [WARN] No pattern found in {img_path.name}, skipping.")
            continue

        if pattern == "chessboard":
            # Refine to sub-pixel accuracy (circle grids are already centroid-precise)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners)
        image_names.append(img_path.name)
        found_count += 1
        print(f"  [OK]   {img_path.name}")

        if debug_dir_path:
            overlay = img.copy()
            cv2.drawChessboardCorners(overlay, board_size, corners, ret)
            cv2.imwrite(str(debug_dir_path / f"corners_{img_path.stem}.png"), overlay)

    if found_count < 3:
        raise RuntimeError(
            f"Need at least 3 usable images; only found {found_count}. "
            "Ensure the checkerboard is fully visible and try more photos."
        )

    print(f"\nCalibrating with {found_count} views ...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # --- Per-image reprojection error ---
    # Re-project each view's known 3-D points through the fitted camera model
    # and compare to where the corners were actually detected. This pinpoints
    # WHICH photos hurt the calibration, instead of just one overall number.
    #
    # NOTE: this must be a per-image RMS (sqrt of mean squared distance) to be
    # in the same units as the overall RMS that cv2.calibrateCamera returns.
    # The common "cv2.norm(...)/N" snippet from OpenCV's own tutorial computes
    # a DIFFERENT, smaller-scale quantity (sqrt(sum of squares)/N instead of
    # sqrt(mean of squares)) — it looks like a per-image error but isn't
    # comparable to the headline RMS, which is misleading when hunting for
    # the images actually responsible for a high overall RMS.
    per_image_errors: list[float] = []
    for i in range(found_count):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        diffs = projected.reshape(-1, 2) - img_points[i].reshape(-1, 2)
        squared_dists = np.sum(diffs ** 2, axis=1)
        err = float(np.sqrt(np.mean(squared_dists)))
        per_image_errors.append(err)

    mean_err = float(np.mean(per_image_errors))
    print("\nPer-image reprojection error (px), worst first:")
    for idx in np.argsort(per_image_errors)[::-1]:
        flag = "  <-- high error, consider removing and re-running" \
            if per_image_errors[idx] > 2 * mean_err else ""
        print(f"  {image_names[idx]:30s} {per_image_errors[idx]:.4f}{flag}")

    if debug_dir_path:
        _save_coverage_map(img_points, image_size, debug_dir_path / "coverage_map.png")
        print(f"\nDebug visuals written to {debug_dir_path}:")
        print("  corners_*.png    — detected points overlaid on each accepted photo")
        print("  coverage_map.png — every detected point pooled across all photos")

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
        "per_image_error_px": dict(zip(image_names, (round(e, 4) for e in per_image_errors))),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(intrinsics, fh, indent=2)

    print(f"\nOverall RMS reprojection error: {rms:.4f} px")
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
                        help="Inner corners (chessboard) or dots (circles) along width")
    parser.add_argument("--rows", type=int, default=6,
                        help="Inner corners (chessboard) or dots (circles) along height")
    parser.add_argument("--square", type=float, default=0.025,
                        help="Spacing between points in metres (default: 0.025 = 2.5 cm). "
                             "MEASURE this directly with a ruler — it scales every distance "
                             "the pipeline ever outputs.")
    parser.add_argument(
        "--pattern", choices=["chessboard", "circles"], default="chessboard",
        help="'chessboard' (default, recommended) or 'circles' for a dot grid"
    )
    parser.add_argument(
        "--out", default="calibration/intrinsics.json",
        help="Output JSON path (default: calibration/intrinsics.json)"
    )
    parser.add_argument(
        "--debug-dir", default=None,
        help="If set, write corner-overlay images and a coverage map here "
             "(e.g. calibration/debug) for visual inspection"
    )
    args = parser.parse_args()

    try:
        rms = calibrate(args.images, args.cols, args.rows, args.square, args.out,
                        pattern=args.pattern, debug_dir=args.debug_dir)
        print(f"\nDone. RMS reprojection error = {rms:.4f} px")
        if rms > 1.0:
            print("WARNING: error > 1.0 px. Consider retaking photos with better coverage.")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
