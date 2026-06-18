"""Sidewalk boundary extraction from a binary mask — Stage B (CPU only).

Approach:
    For each image row that contains at least one sidewalk pixel, find the
    leftmost and rightmost sidewalk pixel columns.  Collect the (column, row)
    pairs for left and right boundaries separately, then fit a low-degree
    polynomial to each set using numpy.polyfit.

    Polynomial form:  u = p(v) = c0 + c1*v + c2*v^2 + ...
        where u = column (horizontal pixel), v = row (vertical pixel).
    We regress u on v (not the usual v on u) because rows with no sidewalk
    pixels are simply skipped, and the sidewalk tends to widen/narrow smoothly
    as a function of row.

Walkable corridor:
    Given the fitted polynomials left_poly(v) and right_poly(v), a pixel (u,v)
    is inside the walkable corridor if:
        left_poly(v) + margin <= u <= right_poly(v) - margin
    where margin (in pixels) adds a safety inset from the raw mask edge.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class SidewalkBoundary:
    """Result of boundary extraction for one frame.

    Attributes:
        left_poly:  1-D numpy polynomial coefficients (highest power first) for
                    the left boundary: u_left = numpy.polyval(left_poly, v)
        right_poly: same for the right boundary.
        valid_rows: int32 array of image rows where sidewalk was detected,
                    i.e. the domain over which the polynomials were fitted.
        poly_degree: polynomial degree used (read from config).
    """
    left_poly: np.ndarray    # shape (degree+1,)
    right_poly: np.ndarray   # shape (degree+1,)
    valid_rows: np.ndarray   # shape (M,), int32
    poly_degree: int


def extract_boundaries(
    mask: np.ndarray,
    poly_degree: int = 2,
    min_row_width: int = 5,
) -> SidewalkBoundary | None:
    """Fit left/right boundary polynomials to a binary sidewalk mask.

    Args:
        mask: uint8 array of shape (H, W). Sidewalk pixels have value > 127.
        poly_degree: degree of the polynomial to fit (default 2 = parabola).
        min_row_width: minimum sidewalk width in pixels for a row to be used
                       (filters rows where the mask is just noise).

    Returns:
        SidewalkBoundary if enough rows were found, else ``None``.
    """
    binary = mask > 127  # bool (H, W)
    H, W = mask.shape

    left_us: list[int] = []
    right_us: list[int] = []
    valid_rows: list[int] = []

    for v in range(H):
        row = binary[v]
        cols = np.where(row)[0]
        if len(cols) < min_row_width:
            continue
        left_us.append(int(cols[0]))
        right_us.append(int(cols[-1]))
        valid_rows.append(v)

    if len(valid_rows) < poly_degree + 1:
        return None  # not enough rows to fit the polynomial

    vs = np.array(valid_rows, dtype=np.float64)
    lu = np.array(left_us, dtype=np.float64)
    ru = np.array(right_us, dtype=np.float64)

    left_poly = np.polyfit(vs, lu, poly_degree)
    right_poly = np.polyfit(vs, ru, poly_degree)

    return SidewalkBoundary(
        left_poly=left_poly,
        right_poly=right_poly,
        valid_rows=np.array(valid_rows, dtype=np.int32),
        poly_degree=poly_degree,
    )


def points_in_corridor(
    pixels: np.ndarray,
    boundary: SidewalkBoundary,
    margin: int = 20,
) -> np.ndarray:
    """Return a boolean mask selecting pixels inside the walkable corridor.

    The corridor is the region between left_poly(v)+margin and
    right_poly(v)-margin for each row v.

    Args:
        pixels: int array of shape (N, 2) — pixel (u, v) coordinates.
        boundary: fitted SidewalkBoundary from :func:`extract_boundaries`.
        margin: safety inset in pixels from the raw polynomial boundary.

    Returns:
        inside: bool array of shape (N,), True if the pixel is in the corridor.
    """
    u = pixels[:, 0].astype(np.float64)
    v = pixels[:, 1].astype(np.float64)

    left_raw = np.polyval(boundary.left_poly, v)
    right_raw = np.polyval(boundary.right_poly, v)
    actual_width = np.maximum(0, right_raw - left_raw)
    if margin >= 0:
        safe_margin = np.maximum(0, np.minimum(margin, (actual_width - 2) / 2.0))
    else:
        safe_margin = np.full_like(actual_width, margin)
    
    left_u = left_raw + safe_margin
    right_u = right_raw - safe_margin

    v_min = float(boundary.valid_rows.min())
    v_max = float(boundary.valid_rows.max())
    in_row_range = (v >= v_min) & (v <= v_max)

    return in_row_range & (u >= left_u) & (u <= right_u)


def corridor_mask(
    shape: tuple[int, int],
    boundary: SidewalkBoundary,
    margin: int = 20,
    max_width_px: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize the walkable corridor into a full-resolution boolean mask.

    This restricts a downstream back-projection to a bounded region of the
    frame instead of every pixel — sky, building facades, and anything far
    outside the corridor can never contain a sidewalk obstacle, so excluding
    them up front avoids back-projecting (and clustering) millions of
    irrelevant pixels per frame.

    Args:
        shape: (H, W) of the target frame.
        boundary: fitted SidewalkBoundary from :func:`extract_boundaries`.
        margin: safety inset in pixels from the raw polynomial boundary.
        max_width_px: optional 1-D array of shape (v_max-v_min+1,) giving the
            maximum corridor width in pixels for each row.  When provided the
            right boundary is clamped to ``left_u + max_width_px`` so the
            corridor cannot bleed into the car road.  Compute from camera
            intrinsics as ``max_width_m * fx / Z_row``.

    Returns:
        mask: uint8 array of shape (H, W), 255 inside the corridor band.
    """
    import cv2
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)

    v_min = max(0, int(boundary.valid_rows.min()))
    v_max = min(H - 1, int(boundary.valid_rows.max()))
    if v_max < v_min:
        return mask

    rows = np.arange(v_min, v_max + 1)
    left_raw  = np.polyval(boundary.left_poly,  rows)
    right_raw = np.polyval(boundary.right_poly, rows)
    actual_width = np.maximum(0, right_raw - left_raw)
    if margin >= 0:
        safe_margin = np.maximum(0, np.minimum(margin, (actual_width - 2) / 2.0))
    else:
        safe_margin = np.full_like(actual_width, margin)
    
    left_u  = left_raw + safe_margin
    right_u = right_raw - safe_margin

    # Physical width cap: right boundary cannot exceed left + max allowed width.
    # This prevents the corridor from spanning the car road when the traversable
    # mask (class 0 + 1) is wider than the actual pedestrian path.
    if max_width_px is not None:
        right_u = np.minimum(right_u, left_u + max_width_px)

    left_fill = left_u.astype(np.int32)
    right_fill = right_u.astype(np.int32)
    vs = rows.astype(np.int32)
    
    pts_left = np.stack([left_fill, vs], axis=1)
    pts_right = np.stack([right_fill, vs], axis=1)
    polygon = np.concatenate([pts_left, pts_right[::-1]], axis=0).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [polygon], 255)

    return mask


def draw_boundaries(
    frame: np.ndarray,
    boundary: SidewalkBoundary,
    margin: int = 20,
    colour_left: tuple = (0, 255, 255),
    colour_right: tuple = (255, 0, 255),
    colour_fill: tuple = (0, 200, 0),
    fill_alpha: float = 0.25,
) -> np.ndarray:
    """Render the boundary curves and shaded corridor onto a copy of *frame*.

    Args:
        frame: uint8 BGR array of shape (H, W, 3).
        boundary: fitted SidewalkBoundary.
        margin: same margin used in :func:`points_in_corridor`.
        colour_left: BGR colour for the left boundary curve.
        colour_right: BGR colour for the right boundary curve.
        colour_fill: BGR colour for the corridor fill.
        fill_alpha: transparency of the corridor fill (0=transparent, 1=opaque).

    Returns:
        Annotated frame copy (uint8 BGR HxWx3).
    """
    import cv2

    out = frame.copy()
    H = frame.shape[0]
    vs = np.arange(int(boundary.valid_rows.min()), int(boundary.valid_rows.max()) + 1)

    left_raw = np.polyval(boundary.left_poly, vs)
    right_raw = np.polyval(boundary.right_poly, vs)
    actual_width = np.maximum(0, right_raw - left_raw)
    if margin >= 0:
        safe_margin = np.maximum(0, np.minimum(margin, (actual_width - 2) / 2.0))
    else:
        safe_margin = np.full_like(actual_width, margin)
    
    left_fill = (left_raw + safe_margin).astype(np.int32)
    right_fill = (right_raw - safe_margin).astype(np.int32)

    # Shaded corridor polygon
    pts_left = np.stack([left_fill, vs], axis=1)
    pts_right = np.stack([right_fill, vs], axis=1)
    polygon = np.concatenate([pts_left, pts_right[::-1]], axis=0).reshape(-1, 1, 2)
    overlay = out.copy()
    cv2.fillPoly(overlay, [polygon], colour_fill)
    cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

    # Boundary curves
    left_us = left_raw.astype(np.int32)
    right_us = right_raw.astype(np.int32)
    pts_l = np.stack([left_us, vs], axis=1).reshape(-1, 1, 2)
    pts_r = np.stack([right_us, vs], axis=1).reshape(-1, 1, 2)
    cv2.polylines(out, [pts_l], False, colour_left, 2)
    cv2.polylines(out, [pts_r], False, colour_right, 2)

    return out


if __name__ == "__main__":
    import cv2
    import sys

    parser = argparse.ArgumentParser(description="Extract sidewalk boundaries from a mask")
    parser.add_argument("mask_png", help="Binary sidewalk mask .png (255=sidewalk)")
    parser.add_argument("--frame", default=None,
                        help="Optional RGB frame .png to draw boundaries on")
    parser.add_argument("--out", default="boundary_preview.png")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--margin", type=int, default=20)
    args = parser.parse_args()

    mask = cv2.imread(args.mask_png, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERROR: cannot read {args.mask_png}", file=sys.stderr)
        sys.exit(1)

    boundary = extract_boundaries(mask, poly_degree=args.degree)
    if boundary is None:
        print("ERROR: could not extract boundaries (mask too sparse).", file=sys.stderr)
        sys.exit(1)

    print(f"Left poly  (degree {args.degree}): {boundary.left_poly}")
    print(f"Right poly (degree {args.degree}): {boundary.right_poly}")
    print(f"Valid rows: {len(boundary.valid_rows)}")

    if args.frame:
        base = cv2.imread(args.frame)
    else:
        base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    result = draw_boundaries(base, boundary, margin=args.margin)
    cv2.imwrite(args.out, result)
    print(f"Boundary overlay saved to {args.out}")
