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
from src.geometry.ground_plane import fit_ground_plane, point_height_above_plane
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

        # Grid cache: the (H,W) int32 meshgrids built inside backproject are
        # re-used every frame instead of being allocated and freed per call.
        self._grid_cache: dict = {}

        cor_cfg = config["corridor"]
        self._boundary_ema_alpha: float = float(cor_cfg.get("boundary_ema_alpha", 0.3))
        self._max_corridor_width_m: float = float(cor_cfg.get("max_corridor_width_m", 4.0))
        self._max_boundary_jump_px: float = float(
            cor_cfg.get("max_boundary_jump_px", 120)
        )
        # Minimum fraction of pixels that must be set in the sidewalk-only mask
        # (class 1) for it to be used for boundary extraction.  Below this the
        # mask is likely capturing only a thin curb strip or median rather than
        # the actual walking surface, so we fall back to the combined mask +
        # depth-aware width cap.  5 % is insufficient; 15 % is a safe lower bound
        # for a genuinely detected sidewalk.
        self._min_boundary_coverage: float = float(
            cor_cfg.get("min_boundary_coverage", 0.15)
        )
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

        # Auto-scale intrinsics based on the runtime resolution.
        # This perfectly handles converted 540p caches combined with 1080p intrinsics.
        H, W = depth_m.shape
        scale_x = W / float(self._intrinsics.get("width", 1920))
        scale_y = H / float(self._intrinsics.get("height", 1080))
        runtime_intrinsics = {
            **self._intrinsics,
            "fx": self._intrinsics["fx"] * scale_x,
            "fy": self._intrinsics["fy"] * scale_y,
            "cx": self._intrinsics["cx"] * scale_x,
            "cy": self._intrinsics["cy"] * scale_y,
        }
        
        # Apply metric depth scale correction for monocular depth models
        depth_scale = float(self._cfg.get("camera", {}).get("metric_scale_factor", 1.0))
        depth_m = depth_m * depth_scale

        # Step 1 — back-project to 3-D using the full traversable mask (class 0+1).
        # Keep ground_pixels for height-based boundary filtering in Step 3.
        points, ground_pixels = backproject(depth_m, runtime_intrinsics, mask=mask,
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

        # Step 3 — extract sidewalk boundaries using height-above-plane filtering.
        #
        # SegFormer labels both the brick sidewalk and the asphalt road as class-0
        # ("road") and also labels the thin concrete curb between them the same way,
        # so there is no gap in the mask to separate them via connected components.
        # The right boundary polynomial then fits to the road's far edge.
        #
        # Fix: the RANSAC ground plane (fitted with a 5 cm inlier threshold) is
        # already the BRICK surface — road pixels at ~15 cm below are 3× outside
        # the inlier band and are treated as outliers by RANSAC.  We reuse the 3-D
        # points from Step 1 to compute each pixel's signed height above the brick
        # plane and rebuild the boundary mask from only those pixels within 10 cm
        # of that plane.  Road pixels (−15 cm) are excluded; brick (~0 cm) and
        # above-ground vegetation (>0 cm) are kept.
        #
        # Falls back to the full combined mask when no plane was fitted yet
        # (first frame or too few inliers).
        if plane is not None and len(points) > 0:
            heights = point_height_above_plane(points, plane)
            near_ground = heights > -0.10   # keep brick (≈0 cm), drop road (≈−15 cm)
            mask_for_boundary = np.zeros_like(mask)
            good_pix = ground_pixels[near_ground]  # (u=col, v=row) per backproject contract
            if len(good_pix) > 0:
                mask_for_boundary[good_pix[:, 1], good_pix[:, 0]] = 255
        else:
            mask_for_boundary = mask

        boundary: SidewalkBoundary | None = extract_boundaries(
            mask_for_boundary,
            poly_degree=cor_cfg["boundary_poly_degree"],
            cx=int(runtime_intrinsics["cx"]),
        )

        # Temporal EMA smoothing: blend detected boundary with previous frame's.
        # This eliminates jitter and "coasts" across gaps (parking entrances,
        # missing curb sections) where boundary detection momentarily fails.
        #
        # Jump rejection: if the new boundary has moved more than max_boundary_jump_px
        # pixels at the mid-row relative to the previous frame, the mask is likely
        # distorted by a transient (car driving past, parking entrance, shadow).
        # In that case, discard the new observation and coast on the previous boundary
        # rather than blending a wildly wrong polynomial into the EMA state.
        if boundary is not None and self._prev_boundary is not None:
            mid_row = float(np.median(boundary.valid_rows))
            right_jump = abs(
                np.polyval(boundary.right_poly, mid_row)
                - np.polyval(self._prev_boundary.right_poly, mid_row)
            )
            left_jump = abs(
                np.polyval(boundary.left_poly, mid_row)
                - np.polyval(self._prev_boundary.left_poly, mid_row)
            )
            if right_jump > self._max_boundary_jump_px or left_jump > self._max_boundary_jump_px:
                boundary = self._prev_boundary  # coast — ignore this frame's boundary

        if boundary is not None:
            if self._prev_boundary is not None and boundary is not self._prev_boundary:
                alpha = self._boundary_ema_alpha
                boundary = SidewalkBoundary(
                    left_poly=(alpha * boundary.left_poly
                                + (1.0 - alpha) * self._prev_boundary.left_poly),
                    right_poly=(alpha * boundary.right_poly
                                 + (1.0 - alpha) * self._prev_boundary.right_poly),
                    valid_rows=boundary.valid_rows,
                    poly_degree=boundary.poly_degree,
                )
            if boundary is not self._prev_boundary:
                self._prev_boundary = boundary
        else:
            # No boundary this frame — coast on the last known good one
            boundary = self._prev_boundary

        # Step 4 — depth-aware corridor width cap (per-row, applied at render time).
        #
        # For each image row, derive the ground-plane depth Z_row and compute
        # max_width_px = max_corridor_width_m * fx / Z_row.
        #
        # Key design choice — Z_floor_m (configurable, default 5 m):
        #   Z_row is clipped to [Z_floor_m, 50 m] instead of [1 m, 50 m].
        #   At Z_row = 1.5 m (near rows), the uncapped width is 2.5*780/1.5 = 1300 px,
        #   which exceeds the image width and the cap becomes inactive — the corridor
        #   then spans the full car road at the bottom of the frame.
        #   Clipping Z_row to 5 m gives max_width_px = 2.5*780/5 = 390 px at ALL rows,
        #   so the cap is always active.  Physically this bounds the displayed corridor
        #   to 2.5 m regardless of how close to the camera that row is.
        #
        # IMPORTANT: this cap is applied at RENDER and SEARCH time via the
        # max_width_px parameter of draw_boundaries() and corridor_mask().
        # We do NOT refit polynomial coefficients to the capped data — refitting
        # a degree-2 polynomial to partially-collapsed data causes extrapolation
        # overshoot that makes boundaries cross each other at extreme rows.
        z_floor_m = float(cor_cfg.get("z_floor_m", 5.0))
        max_width_px: np.ndarray | None = None
        if plane is not None and boundary is not None:
            _, b, c, d = plane
            fy = runtime_intrinsics["fy"]
            fx = runtime_intrinsics["fx"]
            cy = runtime_intrinsics["cy"]
            v_min = max(0, int(boundary.valid_rows.min()))
            v_max = min(depth_m.shape[0] - 1, int(boundary.valid_rows.max()))
            rows_f = np.arange(v_min, v_max + 1, dtype=np.float64)
            # Ground-plane depth at the centre column for each row:
            #   plane eq on camera ray → Z = -d / (b*(v-cy)/fy + c)
            denom = b * (rows_f - cy) / fy + c
            Z_row = np.where(np.abs(denom) > 1e-6, -d / denom, 50.0)
            Z_row = np.clip(Z_row, z_floor_m, 50.0)   # floor = 5 m so cap is always active
            max_width_px = self._max_corridor_width_m * fx / Z_row

        # Step 5 — no polynomial refitting.
        #
        # The depth-aware cap (max_width_px) is applied at render and search time
        # directly inside draw_boundaries() and corridor_mask() via their
        # max_width_px parameter.  Do NOT refit polynomial coefficients to capped
        # column data: a degree-2 polynomial fitted to partially-collapsed values
        # (right = left at rows where the raw right is inside the cap) can
        # extrapolate wildly outside the fitted domain, causing the right and left
        # boundary lines to cross each other at near rows (visible as the green
        # corridor collapsing to zero width or the magenta line diving left).
        visual_boundary = boundary  # same polynomial; cap applied at draw time

        # Step 6 — detect obstacle candidates.
        # The corridor_mask is generated from the EMA boundary + depth-aware cap,
        # so the search area matches the displayed green corridor exactly.
        # Obstacles detected outside the green corridor (on the car road) are
        # excluded without needing a separate uncapped boundary.
        cx_px = runtime_intrinsics["cx"]
        if plane is not None:
            if boundary is not None:
                search_mask = corridor_mask(
                    depth_m.shape, boundary,
                    margin=10,
                    max_width_px=max_width_px,
                    cx_px=cx_px,   # centre on camera direction, not mask edges
                )
            else:
                search_mask = None
            all_points, all_pixels = backproject(depth_m, runtime_intrinsics, mask=search_mask,
                                                  _grid_cache=self._grid_cache)
            obstacles = detect_obstacles(
                all_points,
                all_pixels,
                plane,
                boundary=None,        # search_mask already filters; no double-margin
                height_threshold=obs_cfg["height_threshold"],
                dbscan_eps=obs_cfg["dbscan_eps"],
                dbscan_min_samples=obs_cfg["dbscan_min_samples"],
                min_cluster_size=obs_cfg["min_cluster_size"],
                corridor_margin=cor_cfg["corridor_margin"],
                max_candidate_points=obs_cfg["max_candidate_points"],
                max_obstacle_distance_m=float(obs_cfg.get("max_obstacle_distance_m", 25.0)),
            )

            if frame_index is not None and frame_index % 50 == 0:
                heights_diag = point_height_above_plane(all_points, plane)
                n_above = int((heights_diag > obs_cfg["height_threshold"]).sum())
                print(
                    f"  [obs diag frame {frame_index:05d}] "
                    f"search_pts={len(all_points):,}  "
                    f"above_{obs_cfg['height_threshold']}m={n_above:,}  "
                    f"obstacles={len(obstacles)}"
                )
        else:
            obstacles = []

        # Step 7 — update tracker
        active_tracks = self._tracker.update(obstacles)

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

        # Step 8 — render overlay.
        # Pass max_width_px so draw_boundaries() clamps the right boundary line and
        # fill to the same cap as the obstacle search area.
        out_cfg = self._cfg.get("output", {})
        annotated = render_overlay(
            frame,
            active_tracks,
            boundary=visual_boundary,
            corridor_margin=cor_cfg["corridor_margin"],
            frame_index=frame_index,
            max_display_obstacles=int(out_cfg.get("max_display_obstacles", 8)),
            max_width_px=max_width_px,
            cx_px=cx_px,
        )

        return annotated, active_tracks
