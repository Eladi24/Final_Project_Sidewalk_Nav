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


# Proximity colour thresholds (metres)
_DANGER_DIST = 1.5
_CAUTION_DIST = 3.0

_COLOUR_DANGER = (0, 0, 220)     # BGR red
_COLOUR_CAUTION = (0, 200, 255)  # BGR yellow
_COLOUR_SAFE = (0, 200, 0)       # BGR green


def _obstacle_colour(distance_m: float) -> tuple[int, int, int]:
    if distance_m < _DANGER_DIST:
        return _COLOUR_DANGER
    if distance_m < _CAUTION_DIST:
        return _COLOUR_CAUTION
    return _COLOUR_SAFE


def render_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    boundary: SidewalkBoundary | None = None,
    corridor_margin: int = 20,
    frame_index: int | None = None,
) -> np.ndarray:
    """Render the full annotated overlay onto a copy of *frame*.

    Args:
        frame: uint8 BGR array of shape (H, W, 3).
        tracks: active Track objects from ObstacleTracker.update().
        boundary: fitted SidewalkBoundary, or None to skip corridor drawing.
        corridor_margin: pixel margin for corridor shading.
        frame_index: optional frame number shown in HUD.

    Returns:
        Annotated frame copy (uint8 BGR HxWx3).
    """
    out = frame.copy()

    # --- Sidewalk corridor ---
    if boundary is not None:
        out = draw_boundaries(out, boundary, margin=corridor_margin)

    # --- Obstacle boxes and labels ---
    for track in tracks:
        colour = _obstacle_colour(track.distance_m)
        u_min, v_min, u_max, v_max = track.bbox_px
        cv2.rectangle(out, (u_min, v_min), (u_max, v_max), colour, 2)

        side = "R" if track.bearing_deg > 0 else "L"
        label = f"{track.distance_m:.1f}m {abs(track.bearing_deg):.0f}deg {side}"
        text_y = max(v_min - 8, 14)
        cv2.putText(
            out, label, (u_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA,
        )

    # --- HUD ---
    hud_parts: list[str] = []
    if frame_index is not None:
        hud_parts.append(f"frame {frame_index:05d}")
    hud_parts.append(f"{len(tracks)} obstacle(s)")
    cv2.putText(
        out, "  |  ".join(hud_parts), (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.geometry.backprojection import backproject
    from src.geometry.ground_plane import fit_ground_plane
    from src.segmentation.boundary import extract_boundaries
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

    points, pix = backproject(depth, K, mask=mask)
    plane, _ = fit_ground_plane(points, **cfg["ground_plane"])
    boundary = extract_boundaries(mask, poly_degree=cfg["corridor"]["boundary_poly_degree"])

    obs_cfg = cfg["obstacles"]
    if plane is not None:
        obstacles = detect_obstacles(
            points, pix, plane, boundary,
            height_threshold=obs_cfg["height_threshold"],
            dbscan_eps=obs_cfg["dbscan_eps"],
            dbscan_min_samples=obs_cfg["dbscan_min_samples"],
            min_cluster_size=obs_cfg["min_cluster_size"],
            corridor_margin=cfg["corridor"]["corridor_margin"],
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
