# CLAUDE.md

Guidance for Claude Code working in this repository. Read this fully before
writing or modifying code.

---

## 1. What this project is

**Visual Navigation for the Legally Blind** — monocular sidewalk detection and
obstacle avoidance using single-camera depth estimation.

This is a final project for a university course on *Navigation and Location
Estimation Algorithms*. The grade depends on three things, so optimize for them
in every decision:

1. **Deep understanding of the underlying algorithms.** The code must be
   readable and the non-trivial math (back-projection, RANSAC, DBSCAN, camera
   calibration) must be implemented and documented clearly, not hidden behind a
   library call. Wrapping pretrained networks is fine; the geometry is where the
   original work and the course content live.
2. **An easy-to-build GitHub repo.** Someone must be able to clone it and run a
   demo on a sample clip with a single command, on a machine with no GPU. Cache
   precomputed network outputs so the geometry pipeline runs CPU-only.
3. **A clear in-class presentation.** Every module should be independently
   runnable and produce a visualization of its own output, so figures for the
   slides come for free.

### The goal in one sentence

Given a single forward-facing camera video of someone walking on a sidewalk,
detect the sidewalk boundaries and flag obstacles ahead with their real-world
distance and bearing, then surface that as an annotated video overlay and
optional spoken alerts.

---

## 2. Hardware reality and the hybrid architecture (IMPORTANT)

The developer's local GPU is an **NVIDIA GeForce MX330 (2 GB VRAM)**. This is
**not** enough to run the two transformer models (Depth Anything V2 + SegFormer)
over video. Do not write code that assumes the networks run locally in real
time.

The project is therefore split into two stages with a **file-on-disk contract**
between them:

- **Stage A — neural inference (runs on Google Colab T4, or any CUDA GPU).**
  Runs Depth Anything V2 (metric) and SegFormer per frame. Writes per-frame
  outputs to disk: a metric depth map and a sidewalk mask.
- **Stage B — geometry & logic (runs locally, CPU-only).**
  Reads the cached depth maps and masks, performs back-projection, ground-plane
  fitting, obstacle clustering, tracking, and renders the overlay. This is the
  part the developer iterates on constantly and where the course-relevant code
  lives.

**Why this matters for how you write code:** Stage B must NEVER import torch or
touch a GPU. It depends only on numpy / opencv / open3d / scikit-learn and reads
`.npy` + `.png` files. Keep the dependency boundary clean — if geometry code
starts importing the depth model, that's a bug.

### The data contract between stages

For an input video, Stage A produces a per-frame cache directory:

```
data/cache/<clip_name>/
├── frame_00000.png        # original RGB frame (for overlay rendering)
├── depth_00000.npy        # float32 HxW, metric depth in METERS
├── mask_00000.png         # uint8 HxW, sidewalk segmentation (255 = sidewalk, 0 = not)
├── frame_00001.png
├── depth_00001.npy
├── mask_00001.png
└── ...
```

- Depth is `float32`, units = **meters**, shape matches the frame `(H, W)`.
- Mask is single-channel `uint8`. Treat any value > 127 as sidewalk. (Keep it
  binary for now; multi-class can come later.)
- Frame indices are zero-padded to 5 digits and consistent across the three
  files so the loader can pair them by index.

This contract is the most important thing in the repo. Both stages must agree on
it. If you change it, update this file and the loader in the same commit.

---

## 3. Tech stack

- **Python 3.10+**
- **Stage A (Colab / GPU):** `torch`, `transformers` (SegFormer + optionally
  Depth Anything via HF), `Pillow`, `numpy`
- **Stage B (local / CPU):** `numpy`, `opencv-python`, `open3d`,
  `scikit-learn` (DBSCAN), `pyyaml`
- **Output:** `opencv-python` for overlay video; `pyttsx3` (optional) for audio
- **Calibration:** `opencv-python`

Pin versions loosely in `requirements.txt`. Keep a separate
`requirements-colab.txt` for the GPU-only torch/transformers deps so a grader on
CPU never has to install torch to run Stage B.

---

## 4. The pipeline, stage by stage

Each numbered stage maps to a module under `src/`. The algorithm to understand
and document is named in **bold**.

1. **Depth estimation** (`src/depth/`). Depth Anything V2 *metric* checkpoint
   (outdoor-finetuned, Small or Base) maps an RGB frame to per-pixel depth in
   meters. Use the metric variant, not the relative one — obstacle distances
   must be in real meters. Document the relative-vs-metric distinction and the
   ViT-encoder / DPT-decoder structure in the module docstring.
2. **Sidewalk segmentation** (`src/segmentation/`). SegFormer pretrained on
   Cityscapes; keep the `sidewalk` class. No training required.
3. **Boundary extraction** (`src/segmentation/boundary.py`). From the binary
   mask, find the left/right sidewalk edges and fit a low-order polynomial per
   side to get smooth boundary curves and a "walkable corridor."
4. **Back-projection** (`src/geometry/backprojection.py`). **Pinhole camera
   model.** Each pixel `(u, v)` with depth `Z` becomes a 3D point:
   `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`, `Z = Z`. Requires camera
   intrinsics from calibration.
5. **Ground-plane fitting** (`src/geometry/ground_plane.py`). **RANSAC** plane
   fit on the 3D points inside the sidewalk mask. The inlier plane is the
   ground; its normal gives camera tilt.
6. **Obstacle detection** (`src/obstacles/detector.py`). Points inside the
   walkable corridor that sit more than a height threshold above the fitted
   plane are obstacle candidates. **DBSCAN** clusters them; small clusters are
   discarded as noise. Per cluster, compute distance and bearing.
7. **Tracking** (`src/obstacles/tracker.py`). Temporal smoothing across frames —
   start with an exponential moving average on (distance, bearing); a small
   **Kalman filter** is a strong bonus that ties directly to the course.
8. **Output** (`src/output/`). Annotated overlay video (boundary curves,
   obstacle boxes with distance labels) plus optional spoken alerts.

---

## 5. File-by-file implementation guide

Implement in roughly this order. Each `src/` module should expose a small, clean
API (a class or a couple of functions) AND have an `if __name__ == "__main__":`
block that runs it on a sample input and visualizes the result.

### `configs/default.yaml`
Single source of truth for all tunable values. No magic numbers in code — read
them from here. Include at least:
- paths: model checkpoints, intrinsics file, cache dir, output dir
- camera: path to `intrinsics.json`
- ground_plane: RANSAC distance threshold (m), max iterations, min inliers
- obstacles: height threshold above ground (e.g. 0.15 m), DBSCAN `eps` (m),
  DBSCAN `min_samples`, min cluster size
- corridor: boundary polynomial degree, corridor margin
- tracking: EMA alpha (or Kalman noise params), max frames to keep a lost track
- depth: model name/size, input resolution
- segmentation: model name, sidewalk class id

Provide a `load_config(path)` helper (can live in a small `src/config.py`) that
returns a plain dict or a simple dataclass.

### `calibration/calibrate.py`
**Zhang's checkerboard calibration** via OpenCV. Reads ~20 checkerboard photos,
runs `cv2.findChessboardCorners` + `cv2.calibrateCamera`, writes
`calibration/intrinsics.json` with `fx, fy, cx, cy`, the full camera matrix,
distortion coefficients, and the image resolution they were computed at.
Document why intrinsics are resolution-dependent and must match the footage.
`__main__`: run on the images in a given folder and print reprojection error.

### `src/depth/depth_estimator.py` (Stage A — GPU)
Wraps Depth Anything V2 metric. Class `DepthEstimator` with
`__init__(model_size, device)` and `estimate(rgb_frame) -> depth_meters (float32 HxW)`.
Handle the input resize/normalize the model expects and resize the output back
to the original frame size. Docstring: explain metric vs relative, the
scale-invariant training idea, and that this runs on Colab, not the MX330.
`__main__`: load one image, estimate depth, save a colorized depth PNG for
inspection.

### `src/segmentation/sidewalk_seg.py` (Stage A — GPU)
Wraps SegFormer (Cityscapes). Class `SidewalkSegmenter` with
`segment(rgb_frame) -> binary_mask (uint8 HxW, 255=sidewalk)`. Map the Cityscapes
`sidewalk` class id to the mask. `__main__`: overlay the mask on an image and
save it.

### `src/segmentation/boundary.py` (Stage B — CPU)
Input: binary sidewalk mask. Output: left/right boundary as fitted polynomials
(and a helper to test whether a 3D/2D point is inside the corridor). Approach:
for each image row in the mask, find leftmost and rightmost sidewalk pixels, then
`numpy.polyfit` a low-degree polynomial to each set. Return the polynomials plus
a `points_in_corridor(...)` predicate. `__main__`: draw the fitted boundaries on
the mask and save.

### `src/geometry/backprojection.py` (Stage B — CPU)
Pure pinhole math, no learning. Function
`backproject(depth_meters, intrinsics, mask=None) -> points (N,3)` and ideally a
companion that also returns the pixel coords per point so colors/labels can be
mapped back. Vectorize with numpy (build a `u, v` meshgrid). Document the
equation in the docstring exactly as in section 4. `__main__`: back-project a
cached depth map and visualize the point cloud with Open3D.

### `src/geometry/ground_plane.py` (Stage B — CPU)
**RANSAC** plane fit. Function
`fit_ground_plane(points) -> (plane_coeffs, inlier_mask)` where the plane is
`ax + by + cz + d = 0` with a normalized normal. Implement RANSAC yourself
(sample 3 points, fit plane, count inliers within distance threshold, keep best)
so the algorithm is visible and explainable — do not just call Open3D's
`segment_plane`, though you may cross-check against it in a comment/test.
Document the RANSAC iteration-count formula and why least-squares alone fails
with outliers. Provide `point_height_above_plane(points, plane)`. `__main__`: fit
on a real point cloud, color inliers vs outliers, show in Open3D.

### `src/obstacles/detector.py` (Stage B — CPU)
Ties it together for a single frame. Given points, the ground plane, and the
corridor, select points with height > threshold AND inside the corridor, run
**DBSCAN** (`sklearn.cluster.DBSCAN`), drop clusters below min size, and return a
list of obstacles each with: centroid (3D), nearest distance (m), bearing
(degrees, signed: negative = left, positive = right), and an approximate 2D
bounding box for rendering. `__main__`: run on one cached frame and print the
obstacle list.

### `src/obstacles/tracker.py` (Stage B — CPU)
Temporal smoothing across frames. Start simple: match this frame's obstacles to
existing tracks by nearest centroid, update each track's (distance, bearing) with
an EMA, age out tracks not seen for N frames. Leave a clearly marked extension
point for a Kalman filter on the (distance, bearing) state. Keep the matching
logic readable — this is course-relevant.

### `src/output/overlay.py` (Stage B — CPU)
Draws on the original RGB frame: the fitted boundary curves, a shaded corridor,
and a box + distance label per tracked obstacle (color by proximity). Returns the
annotated frame. `__main__`: render one frame and save.

### `src/output/audio_alerts.py` (Stage B — CPU, optional)
Turns the nearest in-corridor obstacle into a spoken phrase
("obstacle, two meters, slightly left") via `pyttsx3`, rate-limited so it doesn't
talk every frame. Optional for grading but great for the demo.

### `src/pipeline.py` (Stage B — CPU, the orchestrator)
The heart of Stage B. Class `Pipeline(config)` with `process_frame(frame, depth,
mask) -> (annotated_frame, obstacles)` that calls: backprojection → ground plane
→ boundary → detector → tracker → overlay. No torch import here. This is what the
runner scripts drive.

### `scripts/run_video.py`
Entry point for the demo. `python scripts/run_video.py --cache data/cache/<clip>
--config configs/default.yaml --out output/<clip>.mp4`. Loads the cached
frame/depth/mask triplets in order, feeds each to `Pipeline.process_frame`, and
writes the annotated video. Must run with no GPU.

### `scripts/run_inference_colab.py` (or `notebooks/inference.ipynb`) — Stage A
The GPU-side job: take a raw video, decode frames, run `DepthEstimator` and
`SidewalkSegmenter`, and write the cache directory in section 2's format to
Google Drive. This is the ONLY place torch is required. Keep it self-contained so
it can be pasted into Colab. Mount Drive and write outputs there (Colab's local
disk resets on disconnect).

### `scripts/evaluate.py`
Quantitative check against the tape-measure ground truth. For the handful of
frames where real distances were measured, compare predicted vs measured obstacle
distance, report mean/median error and a per-camera scale-correction factor.
This is what elevates the project — implement it even if minimal.

### `notebooks/exploration.ipynb`
Scratchpad for per-stage visual debugging. Not part of the runnable pipeline.

### `tests/`
Lightweight sanity tests for the geometry (it's deterministic and CPU-only, so
easy to test): back-projection round-trips a known synthetic depth/intrinsics
pair; RANSAC recovers a known plane from synthetic points with added outliers;
DBSCAN groups two well-separated synthetic blobs into two clusters. These double
as correctness evidence for the presentation.

---

## 6. Conventions

- **Every `src/` module is independently runnable** via `__main__` and produces a
  visualization of its own output. This is non-negotiable — it's how slide
  figures and debugging both happen.
- **No magic numbers.** All thresholds come from `configs/default.yaml`.
- **Stage B is GPU-free.** Geometry/obstacle/output/pipeline code must not import
  torch or transformers. If a function needs a network output, it reads it from
  the cache.
- **Units are explicit.** Depth and 3D coordinates are in meters; bearings in
  degrees; angles documented. Put units in variable names or docstrings where
  ambiguous (`distance_m`, `bearing_deg`).
- **Docstrings carry the math.** For any module implementing a course algorithm
  (backprojection, RANSAC, calibration, DBSCAN usage, tracking), the docstring
  states the equation/algorithm and why it's used. Assume the reader is grading.
- **Type hints** on public functions. Numpy arrays documented with shape + dtype.
- **Keep functions small and pure** in Stage B so they're testable.
- Prefer vectorized numpy over Python loops in the geometry hot paths.

---

## 7. Data the developer prepares (context, not code)

- 10–20 short phone clips (30–60 s), camera at chest height, walking sidewalks;
  deliberate variety (pedestrians, poles, bins, curbs, shadows, one hard case).
- ~20 checkerboard photos with the **same phone, same resolution, locked lens**
  as the footage, for calibration.
- 5–10 frames with tape-measured real distances to obstacles, for
  `evaluate.py` ground truth.
- `data/` is gitignored except a single small sample clip + its cache, committed
  so the repo is runnable out of the box.

---

## 8. Build & run (target developer experience)

Stage B, no GPU, on the committed sample:

```bash
pip install -r requirements.txt
python scripts/run_video.py --cache data/cache/sample --config configs/default.yaml --out output/sample.mp4
```

Stage A, on Colab (only when processing new footage):

```bash
pip install -r requirements-colab.txt
python scripts/run_inference_colab.py --video data/raw/<clip>.mp4 --out data/cache/<clip>
```

Calibration (once per camera):

```bash
python calibration/calibrate.py --images calibration/images --out calibration/intrinsics.json
```

---

## 9. Known risks to keep in mind while coding

- Monocular metric depth can carry 10–20% scale error. That's why
  `evaluate.py` and the per-camera scale factor exist — surface the error
  honestly rather than hide it.
- Cityscapes-trained segmentation may misfire on sidewalks that differ from its
  training distribution. Handle empty/poor masks gracefully (no crash, skip the
  frame's geometry, keep the video flowing) and treat failure cases as
  presentation material.
- Colab sessions reset their local disk on disconnect — always write the cache
  to mounted Drive in Stage A.

---

## 10. Implementation status (as of initial implementation)

All files have been implemented. Overview:

| File | Status | Notes |
|------|--------|-------|
| `configs/default.yaml` | Done | All tunable values |
| `requirements.txt` | Done | CPU Stage B deps |
| `requirements-colab.txt` | Done | GPU Stage A deps |
| `src/config.py` | Done | `load_config()` helper |
| `calibration/calibrate.py` | Done | Zhang's method via OpenCV |
| `calibration/intrinsics.json` | Done | Placeholder — run calibrate.py to overwrite |
| `src/depth/depth_estimator.py` | Done | Depth Anything V2 metric wrapper |
| `src/segmentation/sidewalk_seg.py` | Done | SegFormer Cityscapes wrapper |
| `src/segmentation/boundary.py` | Done | `polyfit` boundary + corridor predicate |
| `src/geometry/backprojection.py` | Done | Vectorised pinhole back-projection |
| `src/geometry/ground_plane.py` | Done | RANSAC from scratch + height query |
| `src/obstacles/detector.py` | Done | DBSCAN obstacle clustering |
| `src/obstacles/tracker.py` | Done | EMA tracker + Kalman extension stub |
| `src/output/overlay.py` | Done | Corridor + obstacle box overlay |
| `src/output/audio_alerts.py` | Done | pyttsx3 rate-limited TTS |
| `src/pipeline.py` | Done | Stage B orchestrator |
| `scripts/run_video.py` | Done | Demo entry point (CPU only) |
| `scripts/run_inference_colab.py` | Done | Stage A Colab runner |
| `scripts/evaluate.py` | Done | Distance accuracy evaluation |
| `tests/test_geometry.py` | Done | 7 deterministic tests |

### Libraries to install

**Stage B — local CPU virtual environment:**
```
pip install -r requirements.txt
```
Installs: `numpy`, `opencv-python`, `open3d`, `scikit-learn`, `pyyaml`, `pyttsx3`

**Stage A — Google Colab only:**
```
pip install -r requirements-colab.txt
```
Installs: `torch`, `torchvision`, `transformers`, `accelerate`, `Pillow`, `numpy`, `opencv-python`, `pyyaml`

**On Linux, `pyttsx3` also needs:**
```
sudo apt-get install espeak
```