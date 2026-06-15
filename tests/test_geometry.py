"""Geometry correctness tests — all CPU, all deterministic.

These serve dual purpose:
    1. Regression guard: catch accidental breakage of the math.
    2. Presentation evidence: show that the core algorithms work on synthetic
       data where ground truth is known exactly.

Run with:  pytest tests/test_geometry.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.backprojection import backproject
from src.geometry.ground_plane import fit_ground_plane, point_height_above_plane


# ---------------------------------------------------------------------------
# 1. Back-projection round-trip
# ---------------------------------------------------------------------------

class TestBackprojection:
    """Verify the pinhole forward and inverse projection are consistent."""

    def setup_method(self):
        self.intrinsics = {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0}

    def test_known_point(self):
        """A point at (X=0, Y=0, Z=5) should project to the principal point."""
        H, W = 480, 640
        depth = np.zeros((H, W), dtype=np.float32)
        # Principal point pixel: u=cx=320, v=cy=240
        depth[240, 320] = 5.0

        points, pixels = backproject(depth, self.intrinsics)

        assert len(points) == 1
        X, Y, Z = points[0]
        np.testing.assert_allclose(X, 0.0, atol=1e-5)
        np.testing.assert_allclose(Y, 0.0, atol=1e-5)
        np.testing.assert_allclose(Z, 5.0, atol=1e-5)

    def test_roundtrip_projection(self):
        """Back-project then re-project; pixel coordinates should be recovered."""
        H, W = 480, 640
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]

        # Synthetic depth: all pixels have depth = 3.0 m
        depth = np.full((H, W), 3.0, dtype=np.float32)

        points, pixels = backproject(depth, self.intrinsics)
        assert points.shape[1] == 3
        assert pixels.shape[1] == 2

        # Re-project back to pixel space
        u_pred = points[:, 0] * fx / points[:, 2] + cx
        v_pred = points[:, 1] * fy / points[:, 2] + cy

        np.testing.assert_allclose(u_pred, pixels[:, 0].astype(float), atol=1e-3)
        np.testing.assert_allclose(v_pred, pixels[:, 1].astype(float), atol=1e-3)

    def test_mask_filters_pixels(self):
        """Only pixels inside the mask should be back-projected."""
        H, W = 100, 100
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 = 400 pixels

        points, pixels = backproject(depth, self.intrinsics, mask=mask)
        assert len(points) == 400

    def test_zero_depth_excluded(self):
        """Pixels with depth == 0 must be excluded regardless of mask."""
        H, W = 50, 50
        depth = np.zeros((H, W), dtype=np.float32)
        depth[25, 25] = 1.0  # only one valid pixel

        points, pixels = backproject(depth, self.intrinsics)
        assert len(points) == 1


# ---------------------------------------------------------------------------
# 2. RANSAC ground-plane recovery
# ---------------------------------------------------------------------------

class TestRANSACGroundPlane:
    """Verify RANSAC recovers a known synthetic plane despite outliers."""

    def _make_plane_points(
        self,
        n_inliers: int,
        n_outliers: int,
        plane: tuple[float, float, float, float],
        noise_std: float = 0.005,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate synthetic points on a plane plus random outlier points."""
        if rng is None:
            rng = np.random.default_rng(42)

        a, b, c, d = plane
        # Sample random (X, Z) and solve for Y using the plane equation
        X = rng.uniform(-5, 5, size=n_inliers)
        Z = rng.uniform(1, 10, size=n_inliers)
        # ax + by + cz + d = 0  =>  y = -(ax + cz + d) / b
        Y = -(a * X + c * Z + d) / b
        Y += rng.normal(0, noise_std, size=n_inliers)

        inlier_pts = np.stack([X, Y, Z], axis=1).astype(np.float64)

        outlier_pts = rng.uniform([-5, -3, 1], [5, 3, 10], size=(n_outliers, 3))
        return np.concatenate([inlier_pts, outlier_pts], axis=0)

    def test_recovers_horizontal_plane(self):
        """RANSAC should recover a near-horizontal ground plane (Y ~ const)."""
        # Y = -1.5 plane: 0*x + 1*y + 0*z + 1.5 = 0
        true_plane = (0.0, 1.0, 0.0, 1.5)
        rng = np.random.default_rng(0)
        points = self._make_plane_points(500, 100, true_plane, rng=rng)

        plane, inlier_mask = fit_ground_plane(
            points, distance_threshold=0.05, max_iterations=500, min_inliers=50, rng=rng
        )

        assert plane is not None, "RANSAC should find a plane"
        # The recovered normal should align with the true normal (a,b,c) = (0,1,0)
        a, b, c, d = plane
        recovered_normal = np.array([a, b, c])
        true_normal = np.array([0.0, 1.0, 0.0])
        # Normals can be flipped — check alignment in either direction
        alignment = abs(np.dot(recovered_normal, true_normal))
        assert alignment > 0.95, f"Normal misaligned: dot={alignment:.3f}"

    def test_inlier_count_dominates(self):
        """With 80% inliers the RANSAC result should have high inlier fraction."""
        true_plane = (0.1, 1.0, 0.05, 2.0)
        rng = np.random.default_rng(1)
        points = self._make_plane_points(800, 200, true_plane, rng=rng)

        plane, inlier_mask = fit_ground_plane(
            points, distance_threshold=0.05, max_iterations=1000, min_inliers=50, rng=rng
        )

        assert plane is not None
        inlier_frac = inlier_mask.sum() / len(points)
        assert inlier_frac > 0.6, f"Expected > 60% inliers, got {100*inlier_frac:.1f}%"

    def test_height_above_plane(self):
        """point_height_above_plane should return ~0 for on-plane points."""
        plane = np.array([0.0, 1.0, 0.0, 0.5])   # Y = -0.5 (unit normal)
        # Points exactly on the plane: y = -0.5 for any (x, z)
        pts = np.array([[0.0, -0.5, 1.0],
                        [2.0, -0.5, 3.0],
                        [-1.0, -0.5, 5.0]], dtype=np.float64)
        heights = point_height_above_plane(pts, plane)
        np.testing.assert_allclose(heights, 0.0, atol=1e-6)

    def test_height_sign(self):
        """Points above the plane should have positive height."""
        plane = np.array([0.0, 1.0, 0.0, 0.5])  # ground at Y = -0.5
        above = np.array([[0.0, 0.0, 1.0]])       # Y=0 > -0.5  => above
        below = np.array([[0.0, -1.0, 1.0]])      # Y=-1 < -0.5 => below
        assert point_height_above_plane(above, plane)[0] > 0
        assert point_height_above_plane(below, plane)[0] < 0


# ---------------------------------------------------------------------------
# 3. DBSCAN — two well-separated blobs
# ---------------------------------------------------------------------------

class TestDBSCANClustering:
    """Verify that DBSCAN separates two synthetic point clouds correctly."""

    def test_two_blobs(self):
        """Two point clouds separated by > eps should produce exactly 2 clusters."""
        from sklearn.cluster import DBSCAN

        rng = np.random.default_rng(42)
        # Blob A centred at (0, 0, 2)
        blob_a = rng.normal([0.0, 0.0, 2.0], 0.05, size=(80, 3))
        # Blob B centred at (3, 0, 2)  — well separated in X
        blob_b = rng.normal([3.0, 0.0, 2.0], 0.05, size=(80, 3))
        points = np.concatenate([blob_a, blob_b], axis=0)

        labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(points)

        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 2, f"Expected 2 clusters, got {len(unique)}"

    def test_noise_excluded(self):
        """Isolated points far from any cluster should get label -1 (noise)."""
        from sklearn.cluster import DBSCAN

        rng = np.random.default_rng(7)
        cluster = rng.normal([0.0, 0.0, 2.0], 0.05, size=(60, 3))
        # Three isolated noise points far from the cluster
        noise = np.array([[10.0, 0.0, 2.0],
                          [0.0, 10.0, 2.0],
                          [0.0, 0.0, 20.0]])
        points = np.concatenate([cluster, noise], axis=0)

        labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(points)

        noise_count = (labels == -1).sum()
        assert noise_count >= 3, f"Expected >= 3 noise points, got {noise_count}"


# ---------------------------------------------------------------------------
# 4. Boundary extraction smoke test
# ---------------------------------------------------------------------------

class TestBoundaryExtraction:
    """Verify polynomial boundary fitting on a synthetic mask."""

    def test_straight_corridor(self):
        """A rectangular sidewalk mask should produce near-vertical boundary polys."""
        from src.segmentation.boundary import extract_boundaries

        H, W = 200, 100
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:, 20:80] = 255  # sidewalk occupies columns 20-79

        boundary = extract_boundaries(mask, poly_degree=1)
        assert boundary is not None

        # Evaluate polys at a middle row
        v_mid = H // 2
        left_u = np.polyval(boundary.left_poly, v_mid)
        right_u = np.polyval(boundary.right_poly, v_mid)

        assert abs(left_u - 20) < 3, f"Left boundary off: {left_u:.1f}"
        assert abs(right_u - 79) < 3, f"Right boundary off: {right_u:.1f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
