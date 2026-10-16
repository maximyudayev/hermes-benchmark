# Video2Motion: HERMES Markerless Biomechanics & Virtual IMU Pipeline

End-to-end pipeline for converting multi-view HERMES video feeds into biomechanically grounded markerless motion trajectories, subject-scaled OpenSim musculoskeletal kinematics, and local virtual IMU signals.

---

## Architecture Overview

```
video2motion/
├── configs/
│   ├── Config.toml                 # Pose2Sim global & trial configuration
│   ├── calibration_points.json     # Static 3D scene landmark coordinates
│   └── opensim_settings/
│       ├── Scaling_Setup_Halpe26.xml # Subject scaling rules for HALPE_26
│       ├── IK_Setup_Halpe26.xml      # OpenSim Inverse Kinematics task weights
│       ├── Markers_Halpe26.xml       # Marker set definition for Rajagopal model
│       └── BodyKinematics_Setup.xml  # Segment acceleration & orientation setup
├── calibration/
│   └── Calib.toml                  # Active Pose2Sim camera calibration
├── data/
│   ├── synchronized/
│   │   └── alignment_manifest.csv  # Harmonized master time-grid mapping
│   └── calibration/
│       ├── intrinsics/             # Camera intrinsic matrices & distortion coefficients
│       └── extrinsics/             # Interactive 2D annotations & reference snapshots
├── models/
│   └── Rajagopal_Pose2Sim.osim     # ISB-compliant musculoskeletal model
├── pose/                           # Multi-view 2D pose JSONs (cam_<id>_json/)
├── pose-3d/                        # Triangulated, filtered, & augmented .trc files
├── kinematics/                     # Scaled .osim models & joint kinematics (.mot)
├── src/
│   ├── 01_sync_aligner.py          # HDF5 timestamp harmonization & grid alignment
│   ├── 02_intrinsic_calibrator.py  # OpenCV ChArUco / checkerboard / nominal solver
│   ├── 03_scene_calibrator.py      # Interactive PnP scene-point extrinsic solver
│   ├── 04_pose_detector.py         # RTMPose HALPE_26 multi-view inference & missing sentinel
│   ├── 05_triangulate_filter.py    # Robust triangulation, Butterworth & LSTM augmentation
│   ├── 06_opensim_ik_runner.py     # Automated scaling & Inverse Kinematics runner
│   └── 07_imu_inertial_engine.py   # Local virtual IMU synthesis & relative kinematics
└── tests/
    ├── test_time_alignment.py      # Master grid integrity & missing-slot assertions
    ├── test_calibration_reproj.py  # PnP camera matrix & reprojection error assertions
    ├── test_triangulation.py       # Triangulation & bone length rigidity check (< 1.5 mm std)
    ├── test_kinematic_bounds.py    # Physiological joint angle boundary assertions
    └── test_imu_physics.py         # Virtual IMU 1g gravity, quaternion & rate tests
```

---

## Environment Prerequisites

The pipeline requires Python with `opensim`, `pose2sim`, `rtmlib`, `opencv-contrib-python`, `h5py`, `scipy`, and `pandas`.
All commands should be executed from the `benchmark/synchronization/video2motion`.

---

## Step-by-Step Pipeline Workflow

### Step 1: Timestamp Harmonization & Grid Alignment (`01_sync_aligner.py`)

Harmonizes asynchronous camera clocks and accounts for dropped frames across multi-view video feeds.

- **Data Ingestion:** Reads camera clock datasets from raw session `cameras.hdf5` (`cameras/<cam_id>/toa_s`, `frame_index`) following the PysioViz schema, paired with raw `.mkv` or `.mp4` video streams. Handles both absolute UNIX timestamps ($\approx 1.76 \times 10^9\text{ s}$) and relative trial offsets.
- **Master Time Grid:** Constructs a uniform discrete time grid at the target sampling rate ($f_{target} = 30.0\text{ Hz}$, interval $\Delta t \approx 33.33\text{ ms}$).
- **Binary Matching & Guarding:** For every grid timestamp $t_k$, maps each camera to its closest frame within tolerance ($\pm 0.5 \Delta t$). If a camera dropped a frame, marks `cam_valid = False` with sentinel indices to prevent inter-camera desynchronization.
- **Reference Snapshots:** Supports extracting synchronized reference video frames for subsequent camera extrinsic calibration (`--extract-snapshots`).

```powershell
& uv run src/01_sync_aligner.py `
  --hdf5 "C:/path/to/raw/trial_0/cameras.hdf5" `
  --videos "C:/path/to/raw/trial_0" `
  --fps 30.0 `
  --start-time 97.0 `
  --duration 30.0 `
  --output-manifest data/synchronized/alignment_manifest_30s.csv `
  --extract-snapshots
```

---

### Step 2: Dual-Stage Camera Calibration (`02_intrinsic_calibrator.py` & `03_scene_calibrator.py`)

Computes intrinsic optical matrices and extrinsic camera poses in a shared world coordinate frame.

- **Stage A — Intrinsics (`02_intrinsic_calibrator.py`):**
  - Solves focal length $(f_x, f_y)$, principal point $(c_x, c_y)$, and radial/tangential lens distortion using ChArUco or checkerboard patterns.
  - Supports nominal intrinsic matrix generation when physical calibration boards are unavailable, deriving parameters from sensor optical dimensions, center crops, and lens focal length (e.g. 4mm fixed lens on 5MP sensor center-cropped to 1440p and scaled to 1080p: $f_x=f_y=2142.86$, $c_x=960.0$, $c_y=540.0$).
- **Stage B — Scene Extrinsics (`03_scene_calibrator.py`):**
  - Uses fixed 3D static scene reference points ($N \ge 6$) measured relative to a defined floor origin (e.g., room tile corners, door posts, static markers).
  - Provides an interactive GUI (`--interactive`) allowing operators to click corresponding 2D landmark coordinates across camera reference snapshots.
  - Solves extrinsic camera rotation vectors $\mathbf{r}$ and translation vectors $\mathbf{t}$ via OpenCV PnP (`cv2.solvePnP` with `SOLVEPNP_ITERATIVE`), compiling the unified multi-camera `Calib.toml`.

```powershell
# Interactive extrinsic scene calibration
& uv run src/03_scene_calibrator.py `
  --intrinsics data/calibration/intrinsics/intrinsics.toml `
  --snapshots-dir data/calibration/extrinsics/snapshots `
  --interactive `
  --output-calib calibration/Calib.toml
```

---

### Step 3: Multi-View 2D Pose Estimation (`04_pose_detector.py`)

Performs deep-learning 2D anatomical landmark inference directly from synchronized multi-camera video streams.

- **Model & Topology:** Utilizes RTMPose with the HALPE_26 skeleton topology (26 keypoints covering head, trunk, upper extremities, and full feet including big toe, small toe, and heel).
- **Direct Video Streaming:** Streams frames on-the-fly directly from raw `.mkv` files at the exact timestamps defined in the alignment manifest, completely avoiding dumping tens of thousands of individual image files to disk.
- **Missing-Frame Sentinel Records:** When a camera drops a frame (`cam_valid == False`), automatically writes OpenPose-compliant dummy JSON records (`person_id: [-1]`, 26 keypoints with confidence `0.0`). This guarantees Pose2Sim never loses lock-step frame indexing between views.
- **Directory Layout:** Outputs OpenPose-format JSONs grouped by camera: `pose/<cam_id>_json/<cam_id>_<frame>_keypoints.json`.

```powershell
& uv run src/04_pose_detector.py `
  --manifest data/synchronized/alignment_manifest_30s.csv `
  --videos-dir "C:/path/to/raw/trial_0" `
  --output-pose-dir pose
```

---

### Step 4: Robust Triangulation & Kinematic Filtering (`05_triangulate_filter.py`)

Reconstructs 3D spatial marker trajectories from multi-view 2D proposals and eliminates tracking jitter.

- **Multi-View 3D Triangulation:** Direct linear transformation (DLT) across active camera views. Cameras exhibiting low detection likelihood ($< 0.3$) or high reprojection errors ($> 150\text{ px}$ given nominal calibration) are progressively excluded per keypoint and frame. Missing values are filled via linear interpolation over small dropout gaps ($\le 20\text{ frames}$).
  - Outputs raw 3D trajectory: `pose-3d/video2motion_0-899.trc`.
- **Zero-Lag Butterworth Low-Pass Filter:** Applies a 4th-order, zero-lag bidirectional Butterworth filter at a 6 Hz cutoff frequency, isolating physiological ADL gait kinematics from high-frequency detector jitter.
  - Outputs filtered trajectory: `pose-3d/video2motion_0-899_filt_butterworth.trc`.
- **Stanford LSTM Marker Augmentation:** Applies a deep recurrent model (Stanford LSTM v0.3) to predict missing anatomical joint centers and occluded landmarks across both upper and lower body segments, expanding the landmark set to 65 biomechanical markers.
  - Outputs augmented trajectory: `pose-3d/video2motion_0-899_filt_butterworth_LSTM.trc`.

```powershell
& uv run src/05_triangulate_filter.py --config Config.toml
```

---

### Step 5: OpenSim Musculoskeletal Model Scaling & Inverse Kinematics (`06_opensim_ik_runner.py`)

Fits an ISB-compliant musculoskeletal model (Rajagopal full-body model) to the 3D marker trajectories to derive physiological joint angles.

#### Execution Modes & Use-Case Guidance

`06_opensim_ik_runner.py` provides two distinct execution modes:

| Scenario / Goal | Recommended Mode | Command |
| :--- | :--- | :--- |
| **New Participant / Baseline Trial**<br>Need to derive subject segment lengths (femur, tibia, pelvis, torso) and scale bone geometry to participant anthropometry. | **Automated Scaling & IK Pipeline** (`--pipeline`) | `& uv run src/06_opensim_ik_runner.py --pipeline` |
| **Subsequent Trials of Same Subject**<br>Maintain constant bone segment lengths across trials (walking, running, perturbations) to prevent anthropometric drift. | **Standalone Direct IK Runner** (Default) | `& uv run src/06_opensim_ik_runner.py --model kinematics/*_scaled.osim` |
| **HERMES Ground-Truth Benchmark**<br>Compare markerless kinematics against Vicon / Xsens on the *exact same* calibrated laboratory model. | **Standalone Direct IK Runner** (Default) | `& uv run src/06_opensim_ik_runner.py --model path/to/vicon_scaled.osim` |
| **Fast Iteration / Task Weight Tuning**<br>Iterate on IK solver marker weights in `IK_Setup_Halpe26.xml` in ~4 seconds without re-scaling. | **Standalone Direct IK Runner** (Default) | `& uv run src/06_opensim_ik_runner.py` |

#### Mode 1: Automated Scaling & IK Pipeline (`--pipeline`)
Dynamically calculates subject segment dimensions, generates a personalized `kinematics/*_scaled.osim` model, and runs Inverse Kinematics:
```powershell
& uv run src/06_opensim_ik_runner.py --pipeline --config Config.toml
```
- **Outputs:**
  - Scaled OpenSim Model: [`kinematics/video2motion_0-899_filt_butterworth_LSTM.osim`](/synchronization/video2motion/kinematics/video2motion_0-899_filt_butterworth_LSTM.osim)
  - Joint Angle Motion File: [`kinematics/video2motion_0-899_filt_butterworth_LSTM.mot`](/synchronization/video2motion/kinematics/video2motion_0-899_filt_butterworth_LSTM.mot)
  - Marker Tracking Error Log: [`kinematics/_ik_marker_errors.sto`](/synchronization/video2motion/kinematics/_ik_marker_errors.sto)

#### Mode 2: Standalone Direct IK Runner
Runs OpenSim `InverseKinematicsTool` in ~4 seconds directly on an existing scaled model:
```powershell
& uv run src/06_opensim_ik_runner.py `
  --model kinematics/video2motion_0-899_filt_butterworth_LSTM.osim `
  --trc pose-3d/video2motion_0-899_filt_butterworth_LSTM.trc `
  --output-mot opensim/Results/trial_kinematics.mot `
  --ik-setup configs/opensim_settings/IK_Setup_Halpe26.xml
```

---

### Step 6: Local Virtual IMU Synthesis & Inter-Segment Kinematics (`07_imu_inertial_engine.py`)

Transforms musculoskeletal joint motions into local synthetic inertial measurement unit (IMU) data streams.

- **Forward Kinematics & Body Acceleration:** Solves OpenSim rigid-body spatial transformations, velocities, and linear accelerations $\mathbf{a}_{global}(t)$ at specific anatomical sensor attachment sites (e.g. pelvis, thighs, shanks, feet, sternum).
- **Technical Coordinate System (TCS) Transformation:** Projects global accelerations and angular velocities into body-fixed local sensor frames ($\mathbf{R}_{B}^{G}(t)$).
- **Specific Force / 1g Gravity Injection:** Synthesizes realistic triaxial accelerometer output by subtracting the gravitational acceleration vector expressed in the local sensor frame:
  $$\mathbf{a}_{meas}^{local}(t) = \mathbf{R}_{G}^{B}(t) \left( \mathbf{a}_{global}(t) - \mathbf{g}_{global} \right)$$
- **Angular Rates & Cardan Kinematics:** Computes local angular velocities $\boldsymbol{\omega}^{local}(t)$ and angular accelerations $\boldsymbol{\alpha}^{local}(t)$, accompanied by unit orientation quaternions $\mathbf{q}(t)$ and relative inter-segment Cardan angles (ISB sequence: flexion/extension, adduction/abduction, internal/external rotation).

```powershell
& uv run src/07_imu_inertial_engine.py `
  --model kinematics/video2motion_0-899_filt_butterworth_LSTM.osim `
  --mot kinematics/video2motion_0-899_filt_butterworth_LSTM.mot `
  --output-csv data/virtual_imu_dataset.csv
```

---

## Validation Test Suite

A comprehensive test suite verifies each mathematical and physical invariant of the pipeline:

```powershell
& uv run -m pytest tests/ -v
```

1. **`tests/test_time_alignment.py`:** Tests master time grid monotonicity, frame interval tolerances ($|\Delta t| < 10^{-4}\text{ s}$), and missing-slot insertion.
2. **`tests/test_calibration_reproj.py`:** Validates PnP camera pose estimation, rotation Rodrigues conversions, and reprojection error thresholds.
3. **`tests/test_triangulation.py`:** Asserts multi-view triangulation accuracy and segment rigidity constancy ($< 1.5\text{ mm}$ standard deviation on rigid limbs).
4. **`tests/test_kinematic_bounds.py`:** Asserts that output Inverse Kinematics joint angles remain within physiological ranges (e.g. knee flexion $0^\circ \le \theta \le 140^\circ$).
5. **`tests/test_imu_physics.py`:** Validates virtual accelerometer baseline gravity ($|\mathbf{a}_{static}| \approx 9.81\text{ m/s}^2$), gyroscope continuity, and quaternion normalization ($||\mathbf{q}|| = 1.0$).
