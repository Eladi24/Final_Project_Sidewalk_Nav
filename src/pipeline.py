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
from src.segmentation.boundary import extract_boundaries


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
    ) -> tuple[np.ndarray, list[Track]]:
        """Run the full Stage B pipeline on one frame.

        Args:
            frame: uint8 BGR array (H, W, 3) — the original RGB frame.
            depth_m: float32 array (H, W) — metric depth in metres.
            mask: uint8 array (H, W) — sidewalk mask (255 = sidewalk).
            frame_index: optional frame number for the HUD overlay.

        Returns:
            annotated_frame: uint8 BGR (H, W, 3) with overlay drawn on it.
            active_tracks: list of Track objects for the current frame.
        """
        gp_cfg = self._cfg["ground_plane"]
        obs_cfg = self._cfg["obstacles"]
        cor_cfg = self._cfg["corridor"]

        # Step 1 — back-project to 3-D using sidewalk mask
        points, _ = backproject(depth_m, self._intrinsics, mask=mask)

        if len(points) < gp_cfg["ransac_min_inliers"]:
            # Not enough depth inside the mask — skip geometry, return plain frame
            return frame.copy(), []

        # Step 2 — fit ground plane (RANSAC)
        plane, _ = fit_ground_plane(
            points,
            distance_threshold=gp_cfg["ransac_distance_threshold"],
            max_iterations=gp_cfg["ransac_max_iterations"],
            min_inliers=gp_cfg["ransac_min_inliers"],
        )

        # Step 3 — extract sidewalk boundary from mask
        boundary = extract_boundaries(
            mask,
            poly_degree=cor_cfg["boundary_poly_degree"],
        )

        # Step 4 — detect obstacle candidates
        if plane is not None:
            # Back-project ALL pixels (not masked) to find obstacles
            all_points, all_pixels = backproject(depth_m, self._intrinsics, mask=None)
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
