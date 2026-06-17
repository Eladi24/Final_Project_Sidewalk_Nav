"""Pinhole camera back-projection — Stage B (CPU only, no torch).

Pinhole camera model:
    A 3-D point P = (X, Y, Z) in camera coordinates projects onto pixel (u, v):

        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Inverting this (given measured depth Z for each pixel):

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z  (the depth measurement itself)

    where:
        fx, fy  — focal lengths in pixels
        cx, cy  — principal point (optical axis intersection) in pixels
        u, v    — pixel column and row (origin at top-left)
        Z       — metric depth from Stage A (metres)

    The result is a 3-D point cloud in the *camera frame*:
        +X  points right
        +Y  points down
        +Z  points forward (into the scene)

    We vectorise over all pixels using numpy meshgrids, so no Python loops.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def backproject(
    depth_m: np.ndarray,
    intrinsics: dict,
    mask: np.ndarray | None = None,
    _grid_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to a 3-D point cloud.

    Args:
        depth_m: float32 (or float16) array of shape (H, W). Depth in metres.
        intrinsics: dict with keys ``fx``, ``fy``, ``cx``, ``cy`` (all floats).
        mask: optional uint8 array of shape (H, W). Only pixels where
              ``mask > 127`` are back-projected.  If ``None``, all valid
              (depth > 0) pixels are used.
        _grid_cache: optional dict for reusing the pixel-coordinate meshgrid
              across calls of the same resolution.  Pass the same dict object
              every call to avoid re-allocating the (H, W) int32 grids each
              time.  The cache is keyed by (H, W) so a resolution change
              triggers a rebuild automatically.

    Returns:
        points: float32 array of shape (N, 3) — 3-D (X, Y, Z) in metres.
        pixels: int32 array of shape (N, 2) — corresponding (u, v) pixel
                coordinates, useful for mapping colours/labels back onto points.
    """
    H, W = depth_m.shape
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    # Pixel-coordinate meshgrids: int32 is sufficient for any resolution
    # (max coord ~65k << 2^31) and uses half the memory of the default int64.
    # With _grid_cache, the two (H,W) int32 arrays are built ONCE per pipeline
    # and reused every frame, eliminating the largest repeated allocation.
    if _grid_cache is not None and _grid_cache.get("shape") == (H, W):
        u_grid = _grid_cache["u"]
        v_grid = _grid_cache["v"]
    else:
        u_grid, v_grid = np.meshgrid(
            np.arange(W, dtype=np.int32),
            np.arange(H, dtype=np.int32),
        )
        if _grid_cache is not None:
            _grid_cache["shape"] = (H, W)
            _grid_cache["u"] = u_grid
            _grid_cache["v"] = v_grid

    # Validity: positive depth, and inside mask (if provided)
    valid = depth_m > 0.0
    if mask is not None:
        valid = valid & (mask > 127)

    u_valid = u_grid[valid].astype(np.float32)   # shape (N,)
    v_valid = v_grid[valid].astype(np.float32)
    Z = depth_m[valid].astype(np.float32)         # handles float16 input too

    X = (u_valid - cx) * Z / fx
    Y = (v_valid - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=1)                        # (N, 3) float32
    pixels = np.stack([u_grid[valid], v_grid[valid]], axis=1)   # (N, 2) int32

    return points, pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back-project a cached depth map and display the point cloud"
    )
    parser.add_argument("depth_npy", help="Path to depth .npy file (float32 HxW metres)")
    parser.add_argument(
        "--intrinsics", default="calibration/intrinsics.json",
        help="Path to intrinsics JSON (default: calibration/intrinsics.json)"
    )
    parser.add_argument(
        "--mask", default=None,
        help="Optional sidewalk mask .png to restrict back-projection"
    )
    args = parser.parse_args()

    depth = np.load(args.depth_npy)

    with open(args.intrinsics) as fh:
        K = json.load(fh)

    mask_arr = None
    if args.mask:
        import cv2
        mask_arr = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    points, pixels = backproject(depth, K, mask=mask_arr)
    print(f"Point cloud shape: {points.shape}")
    print(f"X range: {points[:,0].min():.2f} .. {points[:,0].max():.2f} m")
    print(f"Y range: {points[:,1].min():.2f} .. {points[:,1].max():.2f} m")
    print(f"Z range: {points[:,2].min():.2f} .. {points[:,2].max():.2f} m")

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Colour by depth (blue=near, red=far)
        Z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].ptp() + 1e-6)
        colours = np.stack([Z_norm, np.zeros_like(Z_norm), 1 - Z_norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colours)

        o3d.visualization.draw_geometries([pcd], window_name="Back-projected point cloud")
    except ImportError:
        print("open3d not installed — skipping 3-D visualisation.")
        print("Install with: pip install open3d")
