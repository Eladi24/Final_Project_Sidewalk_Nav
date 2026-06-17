"""Stage B pipeline orchestrator — CPU only, no torch imports allowed.

Wires together all Stage B modules into a single per-frame call:

    backproject  →  ground_plane  →  boundary  →  detector  →  tracker  →  overlay

The Pipeline class is what scripts/run_video.py drives.  Each module keeps its
own state (the tracker persists across frames); the pipeline just threads data
through in the right order.

If any step fails gracefully (e.g. too few inliers for the ground plane, or an
empty mask), the frame is returned with no obstacle annotations — the video
keeps flowing without crashing.
"""
from __future__ import annotations

import json

import numpy as np

from src.geometry.backprojection import backproject
from src.geometry.ground_plane import fit_ground_plane
from src.obstacles.detector import detect_obstacles
from src.obstacles.tracker import ObstacleTracker, Track
from src.output.overlay import render_overlay
from src.segmentation.boundary import SidewalkBoundary, corridor_mask, extract_boundaries


class Pipeline:
    """Full Stage B pipeline for one clip.

    Args:
        config: nested dict from load_config (configs/default.yaml).
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config

        intrinsics_path = config["camera"]["intrinsics"]
        with open(intrinsics_path) as fh:
            self._intrinsics = json.load(fh)

        # Spatial stride: downsample depth/mask/frame before geometry to cut
        # per-frame memory by stride^2. The meshgrid in backproject is the main
        # offender — (H,W) int64 × 2 per call × 2 calls/frame = 66 MB at 1080p.
        # At stride=2 (960×540) that drops to ~16 MB. RANSAC and DBSCAN produce
        # identical results because their distances are metric (metres), not pixels.
        self._stride = int(config.get("performance", {}).get("spatial_stride", 1))
        if self._stride > 1:
            s = float(self._stride)
            self._intrinsics = {
                **self._intrinsics,
                "fx": self._intrinsics["fx"] / s,
                "fy": self._intrinsics["fy"] / s,
                "cx": self._intrinsics["cx"] / s,
                "cy": self._intrinsics["cy"] / s,
            }

        # Grid cache: the (H,W) int32 meshgrids built inside backproject are
        # re-used every frame instead of being allocated and freed per call.
        self._grid_cache: dict = {}

        cor_cfg = config["corridor"]
        self._boundary_ema_alpha: float = float(cor_cfg.get("boundary_ema_alpha", 0.4))
        self._max_corridor_width_m: float = float(cor_cfg.get("max_corridor_width_m", 4.0))
        self._prev_boundary: SidewalkBoundary | None = None  # temporal EMA state

        t_cfg = config["tracking"]
        self._tracker = ObstacleTracker(
            ema_alpha=t_cfg["ema_alpha"],
            max_lost_frames=t_cfg["max_lost_frames"],
            max_match_distance_m=t_cfg["max_match_distance_m"],
        )

    def process_frame(
        self,
        frame: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        frame_index: int | None = None,
        boundary_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[Track]]:
        """Run the full Stage B pipeline on one frame.

        Args:
            frame: uint8 BGR array (H, W, 3) — the original RGB frame.
            depth_m: float32 array (H, W) — metric depth in metres.
            mask: uint8 array (H, W) — combined traversable mask (255 = walkable).
                  Used for RANSAC ground-plane fitting.
            frame_index: optional frame number for the HUD overlay.
            boundary_mask: optional uint8 (H, W) — sidewalk-only mask (class 1).
                  When provided and sufficiently dense, used instead of *mask*
                  for corridor boundary extraction so the corridor does not bleed
                  into the car road.  Falls back to *mask* if None or too sparse.

        Returns:
            annotated_frame: uint8 BGR (H, W, 3) with overlay drawn on it.
            active_tracks: list of Track objects for the current frame.
        """
        gp_cfg = self._cfg["ground_plane"]
        obs_cfg = self._cfg["obstacles"]
        cor_cfg = self._cfg["corridor"]

        # Downsample all inputs if spatial_stride > 1 (intrinsics already scaled in __init__)
        if self._stride > 1:
            s = self._stride
            depth_m = depth_m[::s, ::s]
            mask = mask[::s, ::s]
            frame = frame[::s, ::s]
            if boundary_mask is not None:
                boundary_mask = boundary_mask[::s, ::s]

        # Step 1 — back-project to 3-D using the full traversable mask (class 0+1)
        points, _ = backproject(depth_m, self._intrinsics, mask=mask,
                                _grid_cache=self._grid_cache)

        if len(points) < gp_cfg["ransac_min_inliers"]:
            return frame.copy(), []

        # Step 2 — fit ground plane (RANSAC)
        plane, _ = fit_ground_plane(
            points,
            distance_threshold=gp_cfg["ransac_distance_threshold"],
            max_iterations=gp_cfg["ransac_max_iterations"],
            min_inliers=gp_cfg["ransac_min_inliers"],
            max_input_points=gp_cfg.get("ransac_max_input_points", 15000),
        )

        # Step 3 — extract sidewalk boundary.
        # Prefer the sidewalk-only mask (class 1) for boundaries so the corridor
        # doesn't extend into the car road.  Fall back to the combined mask if
        # the sidewalk-only mask is too sparse (cobblestone classified as road).
        boundary: SidewalkBoundary | None = None
        if boundary_mask is not None:
            boundary = extract_boundaries(boundary_mask, poly_degree=cor_cfg["boundary_poly_degree"])
        if boundary is None:
            boundary = extract_boundaries(mask, poly_degree=cor_cfg["boundary_poly_degree"])

        # Temporal EMA smoothing: blend detected boundary with previous frame's.
        # This eliminates jitter and "coasts" across gaps (parking entrances,
        # missing curb sections) where boundary detection momentarily fails.
        if boundary is not None:
            if self._prev_boundary is not None:
                alpha = self._boundary_ema_alpha
                boundary = SidewalkBoundary(
                    left_poly=(alpha * boundary.left_poly
                                + (1.0 - alpha) * self._prev_boundary.left_poly),
                    right_poly=(alpha * boundary.right_poly
                                 + (1.0 - alpha) * self._prev_boundary.right_poly),
                    valid_rows=boundary.valid_rows,
                    poly_degree=boundary.poly_degree,
                )
            self._prev_boundary = boundary
        else:
            # No boundary this frame — coast on the last known good one
            boundary = self._prev_boundary

        # Step 4 — depth-aware corridor width cap.
        # For each image row, compute max corridor width in pixels from the
        # camera intrinsics and the ground-plane depth at that row.  This caps
        # the right boundary so it cannot exceed left + max_width metres,
        # preventing the corridor from spanning the full car road even when the
        # combined traversable mask (class 0+1) is very wide.
        max_width_px: np.ndarray | None = None
        if plane is not None and boundary is not None:
            _, b, c, d = plane
            fy = self._intrinsics["fy"]
            fx = self._intrinsics["fx"]
            cy = self._intrinsics["cy"]
            v_min = max(0, int(boundary.valid_rows.min()))
            v_max = min(depth_m.shape[0] - 1, int(boundary.valid_rows.max()))
            rows_f = np.arange(v_min, v_max + 1, dtype=np.float64)
            # Ground-plane depth at the centre column for each row:
            #   plane eq on camera ray → Z = -d / (b*(v-cy)/fy + c)
            denom = b * (rows_f - cy) / fy + c
            Z_row = np.where(np.abs(denom) > 1e-6, -d / denom, 50.0)
            Z_row = np.clip(Z_row, 1.0, 50.0)
            max_width_px = self._max_corridor_width_m * fx / Z_row

        # Step 5 — detect obstacle candidates
        if plane is not None:
            if boundary is not None:
                search_mask = corridor_mask(
                    depth_m.shape, boundary,
                    margin=cor_cfg["corridor_margin"],
                    max_width_px=max_width_px,
                )
            else:
                search_mask = None
            all_points, all_pixels = backproject(depth_m, self._intrinsics, mask=search_mask,
                                                  _grid_cache=self._grid_cache)
            obstacles = detect_obstacles(
                all_points,
                all_pixels,
                plane,
                boundary,
                height_threshold=obs_cfg["height_threshold"],
                dbscan_eps=obs_cfg["dbscan_eps"],
                dbscan_min_samples=obs_cfg["dbscan_min_samples"],
                min_cluster_size=obs_cfg["min_cluster_size"],
                corridor_margin=cor_cfg["corridor_margin"],
                max_candidate_points=obs_cfg["max_candidate_points"],
            )
        else:
            obstacles = []

        # Step 5 — update tracker
        active_tracks = self._tracker.update(obstacles)

        # Attach bbox from the matching obstacle to each track (best effort)
        for track in active_tracks:
            if not hasattr(track, "bbox_px"):
                matched = min(
                    obstacles,
                    key=lambda o: abs(o.distance_m - track.distance_m),
                    default=None,
                )
                if matched is not None:
                    track.bbox_px = matched.bbox_px  # type: ignore[attr-defined]
                else:
                    track.bbox_px = (0, 0, 1, 1)  # type: ignore[attr-defined]

        # Step 6 — render overlay
        annotated = render_overlay(
            frame,
            active_tracks,
            boundary=boundary,
            corridor_margin=cor_cfg["corridor_margin"],
            frame_index=frame_index,
        )

        return annotated, active_tracks
