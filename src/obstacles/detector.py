"""Obstacle detection for a single frame — Stage B (CPU only, no torch).

Pipeline for one frame:
    1. Receive the 3-D point cloud (N x 3) and per-point pixel coordinates
       (from backprojection) together with the fitted ground plane and the
       sidewalk boundary.
    2. Compute each point's height above the ground plane.
    3. Keep only points that are (a) above the height threshold AND (b) inside
       the walkable corridor.  These are obstacle candidates.
    4. Run DBSCAN on the candidate points (in 3-D, using metric distances).
    5. Discard clusters smaller than min_cluster_size (noise).
    6. For each surviving cluster compute:
         - centroid_m  (X, Y, Z) in metres, camera frame
         - distance_m  = sqrt(X^2 + Z^2)  (horizontal ground distance)
         - bearing_deg = arctan2(X, Z) * 180/pi
                         negative = left, positive = right
         - bbox_px     bounding box in pixel space (u_min, v_min, u_max, v_max)

DBSCAN overview (Ester et al. 1996):
    A density-based clustering algorithm that does NOT require specifying the
    number of clusters in advance.
        eps         — neighbourhood radius (metres).  Two points are neighbours
                      if their 3-D Euclidean distance < eps.
        min_samples — a point is a *core point* if it has >= min_samples
                      neighbours within eps.
        Core points that are connected form a cluster; non-core points near a
        core point are border points; remaining points are noise (label = -1).
    We use sklearn.cluster.DBSCAN so the implementation is well-tested, but
    the parameters (eps, min_samples) are exposed in configs/default.yaml.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Obstacle:
    """A single detected obstacle cluster.

    Attributes:
        centroid_m:  float32 (3,)  — (X, Y, Z) in metres, camera frame.
        distance_m:  horizontal distance to the obstacle in metres.
        bearing_deg: signed bearing — negative = left, positive = right.
        bbox_px:     pixel bounding box (u_min, v_min, u_max, v_max).
        cluster_id:  DBSCAN cluster label (>= 0).
    """
    centroid_m: np.ndarray       # shape (3,)
    distance_m: float
    bearing_deg: float
    bbox_px: tuple[int, int, int, int]
    cluster_id: int


def detect_obstacles(
    points: np.ndarray,
    pixels: np.ndarray,
    ground_plane: np.ndarray,
    boundary,                         # SidewalkBoundary or None
    height_threshold: float = 0.15,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 10,
    min_cluster_size: int = 15,
    corridor_margin: int = 20,
    max_candidate_points: int = 50_000,
    max_obstacle_distance_m: float = 25.0,
) -> list[Obstacle]:
    """Detect obstacles in a single frame's point cloud.

    Args:
        points: float32 (N, 3) — 3-D points in camera frame (metres).
        pixels: int32 (N, 2) — corresponding pixel (u, v) for each point.
        ground_plane: float64 (4,) — [a, b, c, d] unit-normal plane equation.
        boundary: SidewalkBoundary from boundary.extract_boundaries, or None
                  (if None, skip corridor filtering).
        height_threshold: minimum height above ground (metres) to be a candidate.
        dbscan_eps: DBSCAN neighbourhood radius in metres.
        dbscan_min_samples: DBSCAN core-point density.
        min_cluster_size: discard clusters with fewer points.
        corridor_margin: pixel inset passed to points_in_corridor.
        max_candidate_points: hard cap on candidates passed to DBSCAN. A noisy
            boundary polynomial can occasionally over-extend the corridor on a
            particular frame (sparse/odd mask data → a bad curve fit far outside
            the densely-sampled rows), producing a candidate set much larger
            than any real obstacle scene needs. Real obstacle clusters are
            dense, so randomly subsampling beyond this cap doesn't change which
            clusters get found — it just bounds the worst-case cost regardless
            of why the candidate set got large.

    Returns:
        List of Obstacle objects, sorted by distance (nearest first).
    """
    if len(points) == 0:
        return []

    # --- 1. Height above ground plane ---
    from src.geometry.ground_plane import point_height_above_plane
    heights = point_height_above_plane(points, ground_plane)

    above_ground = heights > height_threshold

    # --- 2. Corridor filter ---
    if boundary is not None:
        from src.segmentation.boundary import points_in_corridor
        in_corridor = points_in_corridor(pixels, boundary, margin=corridor_margin)
    else:
        in_corridor = np.ones(len(points), dtype=bool)

    candidates = above_ground & in_corridor
    if candidates.sum() == 0:
        return []

    cand_points = points[candidates]   # (M, 3)
    cand_pixels = pixels[candidates]   # (M, 2)

    if len(cand_points) > max_candidate_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(cand_points), size=max_candidate_points, replace=False)
        cand_points = cand_points[keep]
        cand_pixels = cand_pixels[keep]

    # --- 3. DBSCAN clustering (in 3-D metric space) ---
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=1)
    labels = db.fit_predict(cand_points)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    obstacles: list[Obstacle] = []
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        if mask.sum() < min_cluster_size:
            continue

        cluster_pts = cand_points[mask]   # (K, 3)
        cluster_pix = cand_pixels[mask]   # (K, 2)

        centroid = cluster_pts.mean(axis=0).astype(np.float32)

        # Report distance/bearing to the NEAREST point of the cluster, not
        # the centroid. When a cluster mixes obstacle surface pixels (near)
        # with background pixels that bled through the depth map (far), the
        # centroid is pulled toward the background, giving a wildly large
        # reported distance. The nearest point is what a pedestrian would
        # physically encounter.
        horiz_dists = np.sqrt(cluster_pts[:, 0] ** 2 + cluster_pts[:, 2] ** 2)
        nearest_point_idx = int(np.argmin(horiz_dists))
        distance_m = float(horiz_dists[nearest_point_idx])
        X_near, _, Z_near = cluster_pts[nearest_point_idx]
        bearing_deg = float(np.degrees(np.arctan2(X_near, Z_near)))

        u_min = int(cluster_pix[:, 0].min())
        v_min = int(cluster_pix[:, 1].min())
        u_max = int(cluster_pix[:, 0].max())
        v_max = int(cluster_pix[:, 1].max())

        obstacles.append(Obstacle(
            centroid_m=centroid,
            distance_m=distance_m,
            bearing_deg=bearing_deg,
            bbox_px=(u_min, v_min, u_max, v_max),
            cluster_id=int(cluster_id),
        ))

    obstacles = [o for o in obstacles if o.distance_m <= max_obstacle_distance_m]
    obstacles.sort(key=lambda o: o.distance_m)
    return obstacles


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import cv2
    from src.geometry.backprojection import backproject
    from src.geometry.ground_plane import fit_ground_plane
    from src.segmentation.boundary import corridor_mask, extract_boundaries
    from src.config import load_config

    parser = argparse.ArgumentParser(description="Detect obstacles in a cached frame")
    parser.add_argument("frame_dir", help="Cache directory containing frame_XXXXX.png, depth_XXXXX.npy, mask_XXXXX.png")
    parser.add_argument("--index", type=int, default=0, help="Frame index (default 0)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--intrinsics", default="calibration/intrinsics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.intrinsics) as fh:
        K = json.load(fh)

    frame_dir = Path(args.frame_dir)
    idx = args.index
    depth = np.load(frame_dir / f"depth_{idx:05d}.npy")
    mask = cv2.imread(str(frame_dir / f"mask_{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(str(frame_dir / f"frame_{idx:05d}.png"))

    # Ground plane is fit on sidewalk-only points (mask=mask).
    points, pix = backproject(depth, K, mask=mask)
    plane, inliers = fit_ground_plane(points, **cfg["ground_plane"])
    if plane is None:
        print("ERROR: could not fit ground plane.")
        sys.exit(1)

    boundary = extract_boundaries(mask, poly_degree=cfg["corridor"]["boundary_poly_degree"])

    # Obstacles are searched in the corridor band, NOT the sidewalk mask —
    # real obstacles (poles, people, bins) are never classified as 'sidewalk',
    # so restricting to that mask would always find zero obstacles.
    if boundary is not None:
        search_mask = corridor_mask(depth.shape, boundary,
                                    margin=cfg["corridor"]["corridor_margin"])
    else:
        search_mask = None
    search_points, search_pix = backproject(depth, K, mask=search_mask)

    obs_cfg = cfg["obstacles"]
    obstacles = detect_obstacles(
        search_points, search_pix, plane, boundary,
        height_threshold=obs_cfg["height_threshold"],
        dbscan_eps=obs_cfg["dbscan_eps"],
        dbscan_min_samples=obs_cfg["dbscan_min_samples"],
        min_cluster_size=obs_cfg["min_cluster_size"],
        corridor_margin=cfg["corridor"]["corridor_margin"],
        max_candidate_points=obs_cfg["max_candidate_points"],
    )

    print(f"Detected {len(obstacles)} obstacle(s):")
    for i, obs in enumerate(obstacles):
        side = "right" if obs.bearing_deg > 0 else "left"
        print(f"  [{i}] dist={obs.distance_m:.2f}m  bearing={obs.bearing_deg:+.1f}deg ({side})")
