"""Annotated frame overlay renderer — Stage B (CPU only, no torch).

Draws on the original RGB frame:
    1. Shaded walkable corridor (semi-transparent green fill + boundary curves).
    2. Per-obstacle bounding box coloured by proximity:
         - Green  : >= 3 m   (safe distance)
         - Yellow : 1.5 – 3 m (caution)
         - Red    : < 1.5 m  (danger)
    3. Distance and bearing label above each box.
    4. Optional HUD showing frame number and total obstacle count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.obstacles.tracker import Track
from src.segmentation.boundary import SidewalkBoundary, draw_boundaries


# Proximity colour thresholds (metres) — in calibrated physical metres.
# With metric_scale_factor=0.3 applied upstream, these are true real-world distances.
_DANGER_DIST = 2.0    # anything within 2 m physical → RED
_CAUTION_DIST = 4.0   # anything within 4 m physical → ORANGE

_COLOUR_DANGER  = (0,   0, 220)   # BGR red
_COLOUR_CAUTION = (0, 127, 255)   # BGR orange
_COLOUR_SAFE    = (0, 200,   0)   # BGR green


def _obstacle_colour(distance_m: float) -> tuple[int, int, int]:
    if distance_m < _DANGER_DIST:
        return _COLOUR_DANGER
    if distance_m < _CAUTION_DIST:
        return _COLOUR_CAUTION
    return _COLOUR_SAFE


def _corridor_fill_colour(
    tracks: list,
    danger_dist: float = _DANGER_DIST,
    caution_dist: float = _CAUTION_DIST,
) -> tuple[int, int, int]:
    """Return the corridor fill colour based on the nearest obstacle distance.

    Green when clear, orange when an obstacle is within caution range, red when
    within danger range.  The corridor itself changing colour gives an immediate
    visual alarm even before the user reads the distance label.
    """
    if not tracks:
        return _COLOUR_SAFE
    nearest = min(t.distance_m for t in tracks)
    if nearest < danger_dist:
        return _COLOUR_DANGER
    if nearest < caution_dist:
        return _COLOUR_CAUTION
    return _COLOUR_SAFE


def render_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    boundary: SidewalkBoundary | None = None,
    corridor_margin: int = 20,
    frame_index: int | None = None,
    max_display_obstacles: int = 8,
    max_width_px: np.ndarray | None = None,
    cx_px: float | None = None,
) -> np.ndarray:
    """Render the full annotated overlay onto a copy of *frame*.

    Args:
        frame: uint8 BGR array of shape (H, W, 3).
        tracks: active Track objects from ObstacleTracker.update().
        boundary: fitted SidewalkBoundary, or None to skip corridor drawing.
        corridor_margin: pixel margin for corridor shading.
        frame_index: optional frame number shown in HUD.
        max_display_obstacles: cap on obstacle boxes drawn; shows the nearest N
            sorted by distance so the most critical ones are always visible.
        max_width_px: optional per-row depth-aware width cap array, same length
            as the row range of *boundary*.  Passed through to draw_boundaries.
        cx_px: camera principal-point column for the current resolution.
            Passed through to draw_boundaries to centre the corridor on the
            camera pointing direction.

    Returns:
        Annotated frame copy (uint8 BGR HxWx3).
    """
    out = frame.copy()

    # --- Sidewalk corridor (fill colour reacts to nearest obstacle) ---
    if boundary is not None:
        fill_colour = _corridor_fill_colour(tracks)
        out = draw_boundaries(out, boundary, margin=corridor_margin,
                              colour_fill=fill_colour,
                              max_width_px=max_width_px, cx_px=cx_px)

    # --- Obstacle boxes and labels (nearest N only) ---
    display_tracks = sorted(tracks, key=lambda t: t.distance_m)[:max_display_obstacles]
    for track in display_tracks:
        colour = _obstacle_colour(track.distance_m)
        u_min, v_min, u_max, v_max = track.bbox_px
        cv2.rectangle(out, (u_min, v_min), (u_max, v_max), colour, 2)

        side = "R" if track.bearing_deg > 0 else "L"
        # Show forward distance Z and lateral distance X separately.
        # bearing = arctan2(X, Z), so X = dist*sin(bearing), Z = dist*cos(bearing).
        bearing_r = np.radians(track.bearing_deg)
        z_fwd = track.distance_m * np.cos(bearing_r)
        x_lat = abs(track.distance_m * np.sin(bearing_r))
        label = f"{z_fwd:.1f}m fwd  {x_lat:.1f}m {side}"
        text_y = max(v_min - 6, 14)

        # Dark background behind each label so it reads over any colour corridor.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            out,
            (u_min - 1, text_y - th - 2),
            (u_min + tw + 2, text_y + 2),
            (0, 0, 0), cv2.FILLED,
        )
        cv2.putText(
            out, label, (u_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA,
        )

    # --- HUD ---
    hud_parts: list[str] = []
    if frame_index is not None:
        hud_parts.append(f"frame {frame_index:05d}")
    hud_parts.append(f"{len(tracks)} obstacle(s)")
    hud_text = "  |  ".join(hud_parts)
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(out, (4, 4), (8 + hw + 2, 28 + hh), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        out, hud_text, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.geometry.backprojection import backproject
    from src.geometry.ground_plane import fit_ground_plane
    from src.segmentation.boundary import corridor_mask, extract_boundaries
    from src.obstacles.detector import detect_obstacles
    from src.obstacles.tracker import ObstacleTracker, Track
    from src.config import load_config
    import json

    parser = argparse.ArgumentParser(description="Render overlay on a cached frame")
    parser.add_argument("frame_dir", help="Cache directory")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--intrinsics", default="calibration/intrinsics.json")
    parser.add_argument("--out", default="overlay_preview.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.intrinsics) as fh:
        K = json.load(fh)

    fdir = Path(args.frame_dir)
    idx = args.index
    depth = np.load(fdir / f"depth_{idx:05d}.npy")
    mask = cv2.imread(str(fdir / f"mask_{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(str(fdir / f"frame_{idx:05d}.png"))

    if frame is None:
        print(f"ERROR: cannot read frame_{idx:05d}.png", file=sys.stderr)
        sys.exit(1)

    # Ground plane is fit on sidewalk-only points (mask=mask).
    points, pix = backproject(depth, K, mask=mask)
    plane, _ = fit_ground_plane(points, **cfg["ground_plane"])
    boundary = extract_boundaries(mask, poly_degree=cfg["corridor"]["boundary_poly_degree"])

    obs_cfg = cfg["obstacles"]
    if plane is not None:
        # Obstacles are searched in the corridor band, NOT the sidewalk mask —
        # real obstacles (poles, people, bins) are never classified as
        # 'sidewalk', so restricting to that mask would always find zero.
        if boundary is not None:
            search_mask = corridor_mask(depth.shape, boundary,
                                        margin=cfg["corridor"]["corridor_margin"])
        else:
            search_mask = None
        search_points, search_pix = backproject(depth, K, mask=search_mask)
        obstacles = detect_obstacles(
            search_points, search_pix, plane, boundary,
            height_threshold=obs_cfg["height_threshold"],
            dbscan_eps=obs_cfg["dbscan_eps"],
            dbscan_min_samples=obs_cfg["dbscan_min_samples"],
            min_cluster_size=obs_cfg["min_cluster_size"],
            corridor_margin=cfg["corridor"]["corridor_margin"],
            max_candidate_points=obs_cfg["max_candidate_points"],
        )
    else:
        obstacles = []

    t_cfg = cfg["tracking"]
    tracker = ObstacleTracker(
        ema_alpha=t_cfg["ema_alpha"],
        max_lost_frames=t_cfg["max_lost_frames"],
        max_match_distance_m=t_cfg["max_match_distance_m"],
    )
    # Dummy track wrapping — for single-frame demo we bypass the EMA
    tracks = []
    for obs in obstacles:
        tracks.append(Track(
            track_id=obs.cluster_id,
            centroid_m=obs.centroid_m,
            distance_m=obs.distance_m,
            bearing_deg=obs.bearing_deg,
        ))
    # Attach bbox to each track (detector knows it; for demo we copy directly)
    for i, obs in enumerate(obstacles):
        tracks[i].bbox_px = obs.bbox_px  # type: ignore[attr-defined]

    result = render_overlay(frame, tracks, boundary,
                            corridor_margin=cfg["corridor"]["corridor_margin"],
                            frame_index=idx)
    cv2.imwrite(args.out, result)
    print(f"Overlay saved to {args.out}")
