"""RANSAC ground-plane fitting — Stage B (CPU only, no torch).

Why least squares alone fails:
    The 3-D point cloud inside the sidewalk mask is not pure ground — it also
    contains obstacle surfaces, depth noise, and segmentation errors.  A global
    least-squares fit would pull the plane toward the outliers, giving a wrong
    normal and height offset.  RANSAC (Random Sample Consensus) robustly
    recovers the dominant plane by ignoring outliers entirely.

RANSAC algorithm (Fischler & Bolles 1981):
    Repeat for N iterations:
        1. Sample 3 random points (minimum to define a plane).
        2. Fit the unique plane through those 3 points:
               normal n = (p2-p1) x (p3-p1),  then normalise |n| = 1
               plane equation: n . (p - p1) = 0  =>  n.x*X + n.y*Y + n.z*Z + d = 0
               where d = -n . p1
        3. Count inliers: points whose perpendicular distance to the plane
               dist = |a*X + b*Y + c*Z + d|   (with |n|=1, so no division needed)
           is less than the distance threshold tau (e.g. 0.05 m).
        4. If this hypothesis has the most inliers so far, save it.
    Return the best hypothesis.

Iteration count formula:
    N = ceil(log(1 - p) / log(1 - (1 - e)^s))
    where:
        p = 0.99  — desired probability of finding a good sample
        e         — expected fraction of outliers (set conservatively to 0.5)
        s = 3     — sample size (3 points for a plane)
    => N = ceil(log(0.01) / log(1 - 0.125)) = ceil(4.605 / 0.133) = 35
    We use 1000 by default for extra robustness on noisy depth.

Plane representation:
    ax + by + cz + d = 0,  with (a,b,c) being a UNIT normal vector.
    Height of a point above the plane (signed):
        h = a*X + b*Y + c*Z + d
    Positive h = above the plane on the side the normal points toward.
    We flip the normal so that the camera origin (0,0,0) has positive height,
    meaning the "above" direction faces the camera.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fit_plane_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Return normalised plane coefficients (a, b, c, d) through three points."""
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return None  # degenerate (collinear points)
    n = n / norm
    d = -float(np.dot(n, p1))
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def fit_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_iterations: int = 1000,
    min_inliers: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit a ground plane to a 3-D point cloud using RANSAC.

    Implements RANSAC from scratch so the algorithm is transparent and
    explainable (do not replace with open3d.segment_plane).

    Args:
        points: float32/float64 array of shape (N, 3) — 3-D points in metres.
        distance_threshold: inlier distance to plane in metres (tau).
        max_iterations: number of RANSAC iterations (N in the formula above).
        min_inliers: minimum inlier count to accept a hypothesis at all.
        rng: optional numpy random Generator for reproducibility.

    Returns:
        plane: float64 array (4,) = [a, b, c, d] with ||(a,b,c)|| = 1,
               or ``None`` if no plane with enough inliers was found.
        inlier_mask: bool array (N,), True for inlier points.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(points)
    if N < 3:
        return None, np.zeros(N, dtype=bool)

    best_plane: np.ndarray | None = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(N, dtype=bool)

    # Preallocate scratch buffers ONCE and reuse them in-place for every
    # iteration below, instead of allocating fresh `dists`/`inlier_mask`
    # arrays on each of up to `max_iterations` passes. With large point
    # clouds (hundreds of thousands of points) and max_iterations=1000,
    # the naive version allocates/frees gigabytes of short-lived arrays per
    # call — harmless on its own, but across hundreds of video frames this
    # rapid alloc/free churn fragments the memory allocator and can drive
    # RSS up until the OS kills the process. Reusing buffers makes this loop
    # allocate nothing after setup, regardless of max_iterations.
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    dists = np.empty(N, dtype=np.float64)
    tmp = np.empty(N, dtype=np.float64)
    inlier_mask = np.empty(N, dtype=bool)

    for _ in range(max_iterations):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]

        plane = _fit_plane_3pts(p1, p2, p3)
        if plane is None:
            continue  # collinear sample, skip

        a, b, c, d = plane
        # Signed distances (unit normal, so no division), computed in-place
        # into the preallocated `dists`/`tmp` buffers — no new arrays.
        np.multiply(x, a, out=dists)
        np.multiply(y, b, out=tmp)
        dists += tmp
        np.multiply(z, c, out=tmp)
        dists += tmp
        dists += d
        np.abs(dists, out=dists)
        np.less(dists, distance_threshold, out=inlier_mask)
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            # inlier_mask is reused next iteration — must copy to keep this one
            best_inlier_mask = inlier_mask.copy()
            best_plane = plane

    if best_plane is None or best_inlier_count < min_inliers:
        return None, best_inlier_mask

    # Refit the plane to ALL inliers with least squares for a more stable result
    inlier_pts = points[best_inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid)
    n = Vt[-1]  # eigenvector corresponding to smallest singular value
    n = n / np.linalg.norm(n)
    d = -float(np.dot(n, centroid))
    best_plane = np.array([n[0], n[1], n[2], d])

    # Ensure normal points toward camera origin (0,0,0): camera should be "above" the plane
    if best_plane[0] * 0 + best_plane[1] * 0 + best_plane[2] * 0 + best_plane[3] < 0:
        best_plane = -best_plane

    return best_plane, best_inlier_mask


def point_height_above_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Compute signed height of each point above the fitted ground plane.

    With a unit normal, the signed distance formula reduces to:
        h = a*X + b*Y + c*Z + d

    Positive values are on the camera side of the plane (above the ground).

    Args:
        points: float array of shape (N, 3).
        plane: float array of shape (4,) = [a, b, c, d], unit normal.

    Returns:
        heights: float32 array of shape (N,), height in metres.
    """
    a, b, c, d = plane
    heights = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d
    return heights.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit ground plane (RANSAC) to a cached point cloud and visualise"
    )
    parser.add_argument("depth_npy", help="Path to depth .npy file (float32 HxW metres)")
    parser.add_argument("--mask", default=None, help="Optional sidewalk mask .png")
    parser.add_argument(
        "--intrinsics", default="calibration/intrinsics.json"
    )
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="RANSAC inlier distance threshold in metres (default 0.05)")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    # Local import to keep Stage B dependency boundary clean
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.geometry.backprojection import backproject

    depth = np.load(args.depth_npy)
    with open(args.intrinsics) as fh:
        K = json.load(fh)

    mask_arr = None
    if args.mask:
        import cv2
        mask_arr = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    points, _ = backproject(depth, K, mask=mask_arr)
    print(f"Fitting plane on {len(points)} points ...")

    plane, inlier_mask = fit_ground_plane(
        points,
        distance_threshold=args.threshold,
        max_iterations=args.iterations,
    )

    if plane is None:
        print("ERROR: could not fit a plane (too few inliers).")
        sys.exit(1)

    a, b, c, d = plane
    inlier_count = inlier_mask.sum()
    print(f"Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"Inliers: {inlier_count} / {len(points)} ({100*inlier_count/len(points):.1f}%)")

    try:
        import open3d as o3d

        colours = np.zeros((len(points), 3))
        colours[inlier_mask] = [0.2, 0.8, 0.2]    # green = ground inliers
        colours[~inlier_mask] = [0.8, 0.2, 0.2]   # red   = outliers / obstacles

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colours)
        o3d.visualization.draw_geometries([pcd], window_name="RANSAC ground plane")
    except ImportError:
        print("open3d not installed — skipping 3-D visualisation.")
