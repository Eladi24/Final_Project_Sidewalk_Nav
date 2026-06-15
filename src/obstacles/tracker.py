"""Temporal obstacle tracking across frames — Stage B (CPU only, no torch).

Why tracking matters:
    Frame-by-frame obstacle detection is noisy — the same physical obstacle
    may appear/disappear for a frame or shift slightly due to depth noise.
    Tracking smooths these fluctuations and assigns a stable identity to each
    obstacle so the overlay doesn't flicker.

Algorithm — greedy nearest-centroid EMA tracker:
    Each frame, match detected obstacles to existing tracks by finding the
    nearest existing track centroid in 3-D (greedy, closest-first).  Tracks
    that receive a match are updated with an exponential moving average (EMA):

        state_new = alpha * observation + (1 - alpha) * state_old

    where alpha in (0, 1] controls responsiveness (high = trust new readings).
    Tracks not matched for max_lost_frames consecutive frames are pruned.
    New detections that did not match any track start a new track.

Extension point — Kalman filter:
    The EMA is a scalar smoother with no velocity model.  A proper Kalman
    filter on the (distance_m, bearing_deg) state with a constant-velocity
    motion model would:
        - predict where the obstacle will be *before* seeing the new frame,
        - correct that prediction with the new measurement,
        - handle missed detections gracefully via the prediction step.
    The state vector would be [dist, dist_dot, bearing, bearing_dot] with
    a simple linear transition matrix.  See the KALMAN_EXTENSION stub below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.obstacles.detector import Obstacle


@dataclass
class Track:
    """A persistent obstacle track across frames.

    Attributes:
        track_id:    unique integer identifier.
        centroid_m:  smoothed 3-D centroid in metres (EMA).
        distance_m:  smoothed horizontal distance in metres.
        bearing_deg: smoothed bearing in degrees.
        frames_lost: consecutive frames this track had no matching detection.
        age:         total frames this track has been alive.
    """
    track_id: int
    centroid_m: np.ndarray   # shape (3,), float32
    distance_m: float
    bearing_deg: float
    frames_lost: int = 0
    age: int = 1

    # --- KALMAN_EXTENSION ---
    # Uncomment and populate these fields to enable a Kalman filter.
    # State: [dist, dist_dot, bearing, bearing_dot]
    # kalman_x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # kalman_P: np.ndarray = field(default_factory=lambda: np.eye(4))


class ObstacleTracker:
    """Greedy nearest-centroid EMA tracker.

    Args:
        ema_alpha: weight given to the new observation (0 < alpha <= 1).
        max_lost_frames: prune tracks absent for this many consecutive frames.
        max_match_distance_m: 3-D distance threshold for matching (metres).
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        max_lost_frames: int = 10,
        max_match_distance_m: float = 2.0,
    ) -> None:
        self._ema_alpha = ema_alpha
        self._max_lost = max_lost_frames
        self._max_match_dist = max_match_distance_m
        self._tracks: list[Track] = []
        self._next_id = 0

    @property
    def active_tracks(self) -> list[Track]:
        """Tracks not yet pruned (including recently lost ones for overlay)."""
        return [t for t in self._tracks if t.frames_lost == 0]

    def update(self, detections: list[Obstacle]) -> list[Track]:
        """Process one frame's detections and return the current active tracks.

        Matching strategy:
            For each unmatched track, find the nearest unmatched detection.
            If within max_match_distance_m, associate them.  Otherwise, age the
            track as lost.  Remaining detections start new tracks.

        Args:
            detections: list of Obstacle from detect_obstacles (may be empty).

        Returns:
            List of currently active (frames_lost == 0) Track objects.
        """
        unmatched_det = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # Build a distance matrix between every track and every detection
        if self._tracks and detections:
            track_centroids = np.array(
                [t.centroid_m for t in self._tracks], dtype=np.float32
            )  # (T, 3)
            det_centroids = np.array(
                [d.centroid_m for d in detections], dtype=np.float32
            )  # (D, 3)
            # Pairwise distances (T, D)
            diff = track_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=2)  # (T, D)
        else:
            dist_mat = np.empty((len(self._tracks), len(detections)))

        # Greedy matching: sort all (track_idx, det_idx) pairs by distance
        if self._tracks and detections:
            pairs = sorted(
                ((dist_mat[ti, di], ti, di)
                 for ti in range(len(self._tracks))
                 for di in range(len(detections))),
                key=lambda x: x[0],
            )
            for dist, ti, di in pairs:
                if ti in matched_track_ids:
                    continue
                if di not in unmatched_det:
                    continue
                if dist > self._max_match_dist:
                    break  # pairs are sorted; remaining are all > threshold
                det = detections[di]
                track = self._tracks[ti]
                alpha = self._ema_alpha
                track.centroid_m = (alpha * det.centroid_m
                                    + (1 - alpha) * track.centroid_m).astype(np.float32)
                track.distance_m = alpha * det.distance_m + (1 - alpha) * track.distance_m
                track.bearing_deg = alpha * det.bearing_deg + (1 - alpha) * track.bearing_deg
                track.frames_lost = 0
                track.age += 1
                matched_track_ids.add(ti)
                unmatched_det.remove(di)

        # Age out unmatched tracks
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.frames_lost += 1

        # Prune lost tracks
        self._tracks = [t for t in self._tracks if t.frames_lost <= self._max_lost]

        # Start new tracks for unmatched detections
        for di in unmatched_det:
            det = detections[di]
            self._tracks.append(Track(
                track_id=self._next_id,
                centroid_m=det.centroid_m.copy(),
                distance_m=det.distance_m,
                bearing_deg=det.bearing_deg,
            ))
            self._next_id += 1

        return self.active_tracks

    # --- KALMAN_EXTENSION ---
    # def _kalman_predict(self, track: Track, dt: float = 1/30) -> None:
    #     """Constant-velocity prediction step."""
    #     F = np.array([[1, dt, 0, 0],
    #                   [0,  1, 0, 0],
    #                   [0,  0, 1, dt],
    #                   [0,  0, 0,  1]])
    #     Q = np.diag([0.01, 0.1, 0.01, 0.1])  # process noise
    #     track.kalman_x = F @ track.kalman_x
    #     track.kalman_P = F @ track.kalman_P @ F.T + Q
    #
    # def _kalman_update(self, track: Track, dist_m: float, bearing_deg: float) -> None:
    #     """Measurement update step."""
    #     H = np.array([[1, 0, 0, 0],
    #                   [0, 0, 1, 0]])
    #     R = np.diag([0.25, 4.0])  # measurement noise (0.5m, 2deg std)
    #     z = np.array([dist_m, bearing_deg])
    #     y = z - H @ track.kalman_x
    #     S = H @ track.kalman_P @ H.T + R
    #     K = track.kalman_P @ H.T @ np.linalg.inv(S)
    #     track.kalman_x = track.kalman_x + K @ y
    #     track.kalman_P = (np.eye(4) - K @ H) @ track.kalman_P
    #     track.distance_m = float(track.kalman_x[0])
    #     track.bearing_deg = float(track.kalman_x[2])
