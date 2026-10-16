This implementation plan is structured for direct import into an agentic IDE workspace (such as Antigravity) to guide supervised implementation, automated testing, and biomechanical validation.

The pipeline is organized into modular phases, with explicit inputs, data contracts, processing logic, OpenSim integrations, and verification tests, incorporating the refined local virtual IMU synthesis formulation, sensor-to-segment offset transformations, generalized speed differentiation, and relative inter-segment kinematics.

---

### Project Architecture & Directory Blueprint
```
clinical_mocap_pipeline/
├── configs/
│   ├── Config.toml                 # Pose2Sim global & trial configuration
│   ├── calibration_points.json     # Static 3D scene point coordinates
│   └── opensim_settings/
│       ├── Scaling_Setup.xml       # Subject scaling parameters
│       ├── IK_Setup.xml            # OpenSim inverse kinematics setup
│       └── BodyKinematics_Setup.xml# Segment acceleration & orientation setup
├── data/
│   ├── raw/
│   │   ├── timestamps/             # HDF5 clock files per camera
│   │   └── videos/                 # Multi-angle genlocked video files
│   ├── synchronized/
│   │   ├── cam_01/                 # Time-gridded frames / synchronized MP4
│   │   └── cam_02/
│   └── calibration/
│       ├── intrinsics/             # Checkerboard/ChArUco images per camera
│       └── extrinsics/             # Scene reference snapshots & Calib.toml
├── models/
│   └── Rajagopal_Pose2Sim.osim     # ISB-compliant musculoskeletal model
├── src/
│   ├── 01_sync_aligner.py          # HDF5 timestamp binning & missing frame insertion
│   ├── 02_intrinsic_calibrator.py  # OpenCV ChArUco/checkerboard solver
│   ├── 03_scene_calibrator.py      # Pose2Sim scene-point extrinsic solver
│   ├── 04_pose_detector.py         # RTMPose Body_with_feet/HALPE_26 inference
│   ├── 05_triangulate_filter.py    # Robust triangulation & Butterworth/LSTM filtering
│   ├── 06_opensim_ik_runner.py     # Automated scaling & inverse kinematics
│   └── 07_imu_inertial_engine.py   # Local virtual IMU synthesis & relative kinematics
└── tests/
    ├── test_time_alignment.py      # Verification of master grid integrity
    ├── test_calibration_reproj.py  # Reprojection error assertion (<2 px)
    └── test_kinematic_bounds.py    # Joint limit & acceleration continuity checks
```
---

### Phase 1: Timestamp Harmonization & Missing-Frame Grid Alignment

#### Objective

Convert variable-length, genlocked video feeds with dropped frames into a unified master time-grid where frame index $N$ corresponds across all cameras.

#### Technical Logic

1. Read the high-performance clock timestamps $t_{c, i}$ from each camera's HDF5 dataset.
2. Determine the recording boundary $[t_{\min}, t_{\max}]$ where at least two cameras overlap.
3. Construct a discrete master timeline grid with sampling period $\Delta t = 1 / f_s$:

$$\mathcal{T} = [t_0, t_0 + \Delta t, t_0 + 2\Delta t, \dots, t_{\max}]$$


4. Map each camera frame to the nearest master grid slot $k$ within tolerance $\epsilon = \Delta t / 2$.
5. When a camera drops a frame at slot $k$, record a missing-frame sentinel (`is_missing = True`).
6. Export:
* A unified frame manifest (`alignment_manifest.csv`) detailing camera status per slot.
* Re-indexed video streams or directory trees containing blank/black placeholder frames to maintain strict frame-index parity for Pose2Sim batch processing.

#### Core Script Structure (`src/01_sync_aligner.py`)

```python
import h5py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

def build_master_timeline(hdf5_paths: dict, nominal_fps: float) -> pd.DataFrame:
    dt = 1.0 / nominal_fps
    camera_data = {}
    
    # 1. Ingest hardware timestamps
    for cam_id, h5_file in hdf5_paths.items():
        with h5py.File(h5_file, 'r') as h5:
            ts = np.array(h5['timestamps']) # seconds
            frame_indices = np.arange(len(ts))
            camera_data[cam_id] = pd.DataFrame({'frame_idx': frame_indices, 'timestamp': ts})
            
    # 2. Derive global temporal overlap
    start_time = max(df['timestamp'].iloc[0] for df in camera_data.values())
    end_time = min(df['timestamp'].iloc[-1] for df in camera_data.values())
    master_ticks = np.arange(start_time, end_time, dt)
    
    # 3. Bin frames to master ticks
    aligned_records = []
    for slot_idx, t_target in enumerate(master_ticks):
        row = {'master_frame': slot_idx, 'timestamp': t_target}
        for cam_id, df in camera_data.items():
            diffs = np.abs(df['timestamp'] - t_target)
            closest_idx = np.argmin(diffs)
            if diffs[closest_idx] < (dt / 2.0):
                row[f'{cam_id}_src_frame'] = int(df['frame_idx'].iloc[closest_idx])
                row[f'{cam_id}_valid'] = True
            else:
                row[f'{cam_id}_src_frame'] = -1
                row[f'{cam_id}_valid'] = False
        aligned_records.append(row)
        
    return pd.DataFrame(aligned_records)

```

---

### Phase 2: Dual-Stage Camera Calibration

#### Objective

Extract intrinsic parameters from physical hardware and solve extrinsic matrices using $\ge 6$ static scene points to construct a valid Pose2Sim `Calib.toml` file.

#### Technical Logic

* **Stage A (Intrinsics):** Process checkerboard or ChArUco calibration sequences captured by each camera. Calculate intrinsic matrix $\mathbf{K}$ and distortion coefficients $\mathbf{D} = [k_1, k_2, p_1, p_2, k_3]$ via `cv2.calibrateCamera`.


* **Stage B (Extrinsics):** Measure $\ge 6$ static, non-coplanar features in the recording space (e.g., room corners, floor grid intersections, stable furniture bounds) in meters relative to an origin frame $\mathcal{G}$.


* Run Pose2Sim's scene calibration mode (`extrinsics_method = 'scene'`) to compute rotation vectors $\mathbf{r}_{\text{vec}}$ and translation vectors $\mathbf{t}_{\text{vec}}$ via Perspective-n-Point (PnP) optimization.

#### Pose2Sim Calibration Configuration (`configs/Config.toml`)

```toml
[project]
project_dir = "."

[calibration]
calibration_type = 'calculate'

[calibration.calculate.intrinsics]
overwrite_intrinsics = false
show_detection_intrinsics = false

[calibration.calculate.extrinsics]
extrinsics_method = 'scene'
# Real-world 3D coordinates in meters [X, Y, Z]
object_coords_3d = [
    [0.00, 0.00, 0.00],  # Origin: Corner of room tile
    [1.20, 0.00, 0.00],  # Floor X-axis marker
    [0.00, 1.80, 0.00],  # Floor Y-axis marker
    [0.00, 0.00, 2.10],  # Door frame top (Z-axis)
    [1.50, 1.80, 0.45],  # Static chair corner
    [2.40, 0.90, 0.00]   # Far wall marker
]

```

#### Execution

Run `Pose2Sim.calibration()` to interactively click corresponding 2D scene points in each camera view. Pose2Sim writes the resulting parameters into `calibration/Calib.toml` formatted according to Aniposelib specifications.

---

### Phase 3: 2D Multi-View Pose Estimation

#### Objective

Run parallel keypoint detection across synchronized views and handle missing-frame indices gracefully.

#### Technical Logic

1. Deploy RTMPose via RTMLib using the `Body_with_feet` or `HALPE_26` model configurations. These provide 26 landmarks, including critical anatomical markers (calcaneus, big toe, small toe) essential for lower-limb kinematic estimation.


2. For frames where a camera has a missing frame (`cam_valid == False`), output an OpenPose-compliant `.json` record containing 26 dummy keypoints with confidence set to $0.0$ and pixel coordinates to $[0.0, 0.0]$.


3. For valid frames, populate coordinates $[x, y]$ alongside detection probabilities $c \in [0.0, 1.0]$.

#### Implementation Snippet

```python
import json
from pathlib import Path

def write_missing_frame_json(output_path: Path, num_keypoints: int = 26):
    """Writes empty detection JSON to prevent frame-indexing desynchronization."""
    # OpenPose BODY_25 / HALPE format: [x, y, confidence, x, y, confidence, ...]
    dummy_keypoints = [0.0, 0.0, 0.0] * num_keypoints
    payload = {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": dummy_keypoints,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": []
        }]
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f)

```

---

### Phase 4: Robust Triangulation & Kinematic Filtering

#### Objective

Triangulate multi-view 2D proposals into continuous 3D coordinate trajectories, filter tracking jitter, and reconstruct missing points.

#### Pose2Sim Triangulation & Filtering Setup (`configs/Config.toml`)

```toml
[triangulation]
reproj_error_threshold_triangulation = 15 # Max pixel reprojection error
likelihood_threshold = 0.35              # Reject low-confidence detections
min_cameras_for_triangulation = 2        # Allows solving even if a camera dropped a frame
interpolation = 'linear'                 # Fill brief dropouts (< 5 frames)
interp_if_additional_cams_active = 2

[filtering]
type = 'butterworth'                     # Zero-lag Butterworth filter
order = 4
cutoff_frequency = 6                     # 6 Hz low-pass cutoff for clinical ADL gait

[markerAugmentation]
make_augmentation = true                 # Run LSTM model to estimate missing occluded markers

```

#### Pipeline Call

```python
from Pose2Sim import Pose2Sim

# 1. Multi-camera triangulation
Pose2Sim.triangulation('configs/Config.toml')

# 2. Kinematic filtering
Pose2Sim.filtering('configs/Config.toml')

# 3. Marker Augmentation for occlusions during ADL (e.g., chair transfers)
Pose2Sim.markerAugmentation('configs/Config.toml')

```

*Deliverable:* Filtered 3D marker coordinates written to `pose-3d/trial_filt_augmented.trc`.

---

### Phase 5: Musculoskeletal Model Scaling & Inverse Kinematics

#### Objective

Fit a subject-scaled OpenSim model to the triangulated `.trc` trajectories and calculate physiological joint angles without joint dislocation or axis crosstalk.

#### Technical Logic

* **Scaling:** OpenSim adjusts segment mass, inertia tensors, and dimensions using static calibration stance marker distances (e.g., inter-ASIS distance for pelvis, lateral epicondyle to lateral malleolus for tibia).

* **Inverse Kinematics (IK):** Minimizes the weighted squared distance between virtual markers $\mathbf{x}_i^{\text{model}}(\mathbf{q})$ and experimental 3D points $\mathbf{x}_i^{\text{exp}}$:

$$\min_{\mathbf{q}} \sum_{i \in \mathcal{M}} w_i \left\Vert{} \mathbf{x}_i^{\text{exp}} - \mathbf{x}_i^{\text{model}}(\mathbf{q}) \right\Vert{}^2$$

#### Automated Execution Script (`src/06_opensim_ik_runner.py`)

```python
import opensim as osim
from pathlib import Path

def execute_opensim_pipeline(model_path: str, trc_path: str, output_mot_path: str):
    # 1. Load scaled musculoskeletal model
    model = osim.Model(model_path)
    model.initSystem()
    
    # 2. Configure Inverse Kinematics Tool
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(trc_path)
    ik_tool.setOutputMotionFileName(output_mot_path)
    
    # Define time range directly from TRC header
    marker_data = osim.MarkerData(trc_path)
    ik_tool.setStartTime(marker_data.getStartFrameTime())
    ik_tool.setEndTime(marker_data.getLastFrameTime())
    
    # Execute IK
    print(f"Running Inverse Kinematics for {trc_path}...")
    ik_tool.run()
    print(f"Kinematics successfully saved to {output_mot_path}")

if __name__ == "__main__":
    execute_opensim_pipeline(
        model_path="models/Rajagopal_Pose2Sim_scaled.osim",
        trc_path="pose-3d/trial_filt_augmented.trc",
        output_mot_path="opensim/Results/trial_kinematics.mot"
    )

```

---

### Phase 6: Local Virtual IMU Synthesis & Inter-Segment Kinematics Engine

#### Objective

Synthesize local tri-axial accelerometer (specific force) and gyroscope (angular velocity) signals directly expressed in the local sensor housing frame ($S$), incorporate custom sensor-to-bone physical mounting offsets ($\mathbf{R}_{S/B}$ and $\mathbf{p}_{S/B}$), calculate inter-segment relative kinematics (such as knee joint rotations independent of laboratory alignment), and output orientation unit quaternions.

#### Mathematical & Biomechanical Formulation

1. **State Realization via Generalized Speeds and Accelerations:**
When OpenSim evaluates motion from an Inverse Kinematics `.mot` file, it supplies coordinate positions $\mathbf{q}(t)$. Calling spatial acceleration queries (`findStationAccelerationInGround`) requires generalized velocities $\mathbf{u}(t) = \dot{\mathbf{q}}(t)$ and accelerations $\dot{\mathbf{u}}(t) = \ddot{\mathbf{q}}(t)$ to be evaluated dynamically. The generalized coordinates are smoothed and differentiated via central finite differences:

$$\mathbf{u}(t_k) \approx \frac{\mathbf{q}(t_{k+1}) - \mathbf{q}(t_{k-1})}{2\,\Delta t}, \quad \dot{\mathbf{u}}(t_k) \approx \frac{\mathbf{u}(t_{k+1}) - \mathbf{u}(t_{k-1})}{2\,\Delta t}$$

These values are injected into each coordinate's `setSpeedValue` prior to invoking `model.realizeAcceleration(state)`.
2. **Sensor Housing Pose Definition:**
A wearable sensor is mounted at an anatomical location $P$ on body $B$ defined by position offset vector $\mathbf{p}_{S/B}$ and orientation transformation matrix $\mathbf{R}_{S/B} \in SO(3)$. Given the bone segment's transformation in the global laboratory reference frame $\mathbf{R}_{B/G}(t)$, the total orientation of the sensor housing in ground is:

$$\mathbf{R}_{S/G}(t) = \mathbf{R}_{B/G}(t) \, \mathbf{R}_{S/B}$$

3. **Local Gyroscope (Angular Velocity):**
The angular velocity of segment $B$ in the ground frame ${}^G\boldsymbol{\omega}^B(t)$ is projected into the local sensor housing axes:

$${}^S\boldsymbol{\omega}(t) = \mathbf{R}_{S/G}(t)^T \, {}^G\boldsymbol{\omega}^B(t) = \mathbf{R}_{S/B}^T \left( \mathbf{R}_{B/G}(t)^T \, {}^G\boldsymbol{\omega}^B(t) \right)$$

4. **Local Accelerometer (Specific Force):**
Physical accelerometers measure specific force, which is the kinematic linear acceleration of station $P$ minus the Earth's gravitational acceleration vector $\mathbf{g} = [0, -9.81, 0]^T \text{ m/s}^2$:

$$\mathbf{a}_{\text{sens}}(t) = \mathbf{R}_{S/G}(t)^T \left( \mathbf{a}_{P/G}(t) - \mathbf{g} \right)$$

where the global linear acceleration $\mathbf{a}_{P/G}(t)$ is computed from the rigid-body kinematic equation:

$$\mathbf{a}_{P/G}(t) = \mathbf{a}_{O_B/G}(t) + {}^G\boldsymbol{\alpha}^B(t) \times \mathbf{r}_{P/O_B}(t) + {}^G\boldsymbol{\omega}^B(t) \times \left( {}^G\boldsymbol{\omega}^B(t) \times \mathbf{r}_{P/O_B}(t) \right)$$

5. **Orientation Quaternions and Inter-Segment Kinematics:**
* **Sensor Quaternions:** Local housing orientations are converted to unit quaternions $\mathbf{q}_S(t) = [q_x, q_y, q_z, q_w]$ matching formats exported by OpenSense and physical IMU sensors.

* **Relative Inter-Segment Rotation:** Rather than calculating tilt relative to room walls, relative joint orientations (e.g., tibia relative to femur) are extracted directly:

$$\mathbf{R}_{\text{distal}/\text{proximal}}(t) = \mathbf{R}_{\text{proximal}/G}(t)^T \, \mathbf{R}_{\text{distal}/G}(t)$$

and decomposed into clinical Cardan angles ($Z$-$X$-$Y$ order: flexion-extension, adduction-abduction, internal-external rotation).

#### Local IMU Synthesis Engine (`src/07_imu_inertial_engine.py`)

```python
import opensim as osim
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def build_virtual_imu_dataset(
    model_path: str,
    mot_path: str,
    sensor_configurations: dict
) -> pd.DataFrame:
    """
    Synthesizes local IMU specific force, angular rate, quaternions,
    and relative inter-segment kinematics.
    
    sensor_configurations format:
    {
        'shank_r_imu': {
            'parent_body': 'tibia_r',
            'pos_offset': [0.02, -0.18, 0.04],   # [x, y, z] meters on bone
            'rot_offset_deg': [0.0, 90.0, 0.0]   # Euler xyz mounting rotation
        },
        'thigh_r_imu': {
            'parent_body': 'femur_r',
            'pos_offset': [0.04, -0.20, 0.02],
            'rot_offset_deg': [0.0, 0.0, 0.0]
        }
    }
    """
    model = osim.Model(model_path)
    state = model.initSystem()
    
    # 1. Ingest coordinates from IK output storage
    motion = osim.Storage(mot_path)
    labels = [motion.getColumnLabels().get(i) for i in range(motion.getColumnLabels().size())]
    time_col = osim.ArrayDouble()
    motion.getTimeColumn(time_col)
    num_frames = time_col.getSize()
    fps = (num_frames - 1) / (time_col.getLast() - time_col.get(0))
    dt = 1.0 / fps
    
    coord_names = labels[1:]
    q_data = np.zeros((num_frames, len(coord_names)))
    for i in range(num_frames):
        row = motion.getStateVector(i).getData()
        for j in range(len(coord_names)):
            q_data[i, j] = row.get(j)
            
    # 2. Derive generalized speeds (u) and accelerations (udot)
    # Degrees to radians conversion for rotational degrees of freedom
    is_rotational = []
    for c_name in coord_names:
        if model.updCoordinateSet().contains(c_name):
            motion_type = model.updCoordinateSet().get(c_name).getMotionType()
            is_rotational.append(motion_type == osim.Coordinate.Rotational)
        else:
            is_rotational.append(False)
            
    q_rad = q_data.copy()
    for col_idx, rot in enumerate(is_rotational):
        if rot:
            q_rad[:, col_idx] = np.radians(q_data[:, col_idx])
            
    u_rad = np.gradient(q_rad, dt, axis=0)
    
    g_global = np.array([0.0, -9.81, 0.0])
    results = []

    for i in range(num_frames):
        t = time_col.get(i)
        
        # 3. Update generalized coordinates and speeds
        for j, c_name in enumerate(coord_names):
            if model.updCoordinateSet().contains(c_name):
                coord = model.updCoordinateSet().get(c_name)
                coord.setValue(state, q_rad[i, j], False)
                coord.setSpeedValue(state, u_rad[i, j])
                
        model.assemble(state)
        model.realizeAcceleration(state)
        
        frame_dict = {'time': t}
        segment_rot_matrices = {}
        
        for sensor_id, cfg in sensor_configurations.items():
            body = model.getBodySet().get(cfg['parent_body'])
            p_offset = osim.Vec3(*cfg['pos_offset'])
            
            # Local sensor mounting rotation R_S/B
            r_sb = R.from_euler('xyz', cfg['rot_offset_deg'], degrees=True).as_matrix()
            
            # Global bone rotation R_B/G
            r_bg_osim = body.getTransformInGround(state).R()
            r_bg = np.array([[r_bg_osim.get(r, c) for c in range(3)] for r in range(3)])
            segment_rot_matrices[cfg['parent_body']] = r_bg
            
            # Total sensor orientation in ground: R_S/G = R_B/G @ R_S/B
            r_sg = r_bg @ r_sb
            
            # --- LOCAL GYROSCOPE (ANGULAR VELOCITY) ---
            omega_g = body.getAngularVelocityInGround(state)
            omega_g_np = np.array([omega_g.get(0), omega_g.get(1), omega_g.get(2)])
            # Project onto local housing axes: omega_S = R_S/G^T @ omega_G
            omega_local = r_sg.T @ omega_g_np
            frame_dict[f'{sensor_id}_gyro_x_dps'] = np.degrees(omega_local[0])
            frame_dict[f'{sensor_id}_gyro_y_dps'] = np.degrees(omega_local[1])
            frame_dict[f'{sensor_id}_gyro_z_dps'] = np.degrees(omega_local[2])
            
            # --- LOCAL ACCELEROMETER (SPECIFIC FORCE) ---
            acc_station_g = body.findStationAccelerationInGround(state, p_offset)
            acc_station_np = np.array([acc_station_g.get(0), acc_station_g.get(1), acc_station_g.get(2)])
            # a_sens = R_S/G^T @ (a_station_g - g)
            acc_local = r_sg.T @ (acc_station_np - g_global)
            frame_dict[f'{sensor_id}_acc_x_g'] = acc_local[0] / 9.81
            frame_dict[f'{sensor_id}_acc_y_g'] = acc_local[1] / 9.81
            frame_dict[f'{sensor_id}_acc_z_g'] = acc_local[2] / 9.81
            
            # --- LOCAL SENSOR QUATERNIONS ---
            q_sens = R.from_matrix(r_sg).as_quat() # [x, y, z, w]
            frame_dict[f'{sensor_id}_quat_x'] = q_sens[0]
            frame_dict[f'{sensor_id}_quat_y'] = q_sens[1]
            frame_dict[f'{sensor_id}_quat_z'] = q_sens[2]
            frame_dict[f'{sensor_id}_quat_w'] = q_sens[3]
            
        # --- RELATIVE INTER-SEGMENT KINEMATICS (KNEE JOINT) ---
        if 'tibia_r' in segment_rot_matrices and 'femur_r' in segment_rot_matrices:
            r_thigh = segment_rot_matrices['femur_r']
            r_shank = segment_rot_matrices['tibia_r']
            # R_rel: Orientation of shank relative to thigh
            r_rel = r_thigh.T @ r_shank
            knee_angles = R.from_matrix(r_rel).as_euler('zxy', degrees=True)
            frame_dict['knee_r_rel_flexion_deg'] = knee_angles[0]
            frame_dict['knee_r_rel_adduction_deg'] = knee_angles[1]
            frame_dict['knee_r_rel_rotation_deg'] = knee_angles[2]
            
        results.append(frame_dict)
        
    return pd.DataFrame(results)

```

---

### Phase 7: Verification, Validation, and Test Suite

| Phase / Component | Target Test File | Assertion / Validation Threshold | Pass Criteria |
| --- | --- | --- | --- |
| **Grid Alignment** | `tests/test_time_alignment.py` | Check delta between consecutive master frames | Absolute time error $< 10^{-4}$ seconds across all channels |
| **Camera Calibration** | `tests/test_calibration_reproj.py` | Reprojection error of known scene landmarks | Mean Reprojection Error $< 2.0$ pixels |
| **Triangulation** | `tests/test_triangulation.py` | Inter-joint bone length variance across trial | Segment length standard deviation $< 1.5$ mm (rigid segment check) |
| **Inverse Kinematics** | `tests/test_kinematic_bounds.py` | OpenSim marker tracking RMSE and max marker error | Marker RMSE $< 1.5$ cm; Max error $< 3.0$ cm |
| **Physiological Limits** | `tests/test_kinematic_bounds.py` | Anatomical coordinate ranges in `.mot` file | Knee flexion between $-5$ and 140 degrees; Ankle between $-40$ and 30 degrees |
| **Virtual IMU Dynamics** | `tests/test_imu_physics.py` | Simulated accelerometer during static baseline | Sensor acceleration magnitude $= 1.00 \pm 0.02$ g; Angular velocity $< 0.5$ deg/s |

---

### Step-by-Step Antigravity Runbook

Implement the tasks sequentially:

1. **Step 1:** Run `src/01_sync_aligner.py` using your HDF5 clock timestamps to create the master manifest and output the synchronized frame sequences. Verify with `tests/test_time_alignment.py`.
2. **Step 2:** Generate the intrinsic matrix via `src/02_intrinsic_calibrator.py` from checkerboard recordings. Measure your 6 static room points and define them in `configs/Config.toml`.

3. **Step 3:** Run `Pose2Sim.calibration()` to click points on the scene and generate `Calib.toml`. Run `tests/test_calibration_reproj.py` to confirm reprojection error is within limits.

4. **Step 4:** Execute 2D pose estimation across synchronized views, outputting empty detection frames for dropped frame slots. Run `Pose2Sim.triangulation()` and `Pose2Sim.filtering()` to generate `pose-3d/trial_filt.trc`.

5. **Step 5:** Run `src/06_opensim_ik_runner.py` to fit the scaled Rajagopal model to the marker data, and inspect the resulting `.mot` file with `tests/test_kinematic_bounds.py`.

6. **Step 6:** Run `src/07_imu_inertial_engine.py` with your custom `sensor_configurations` dictionary to calculate local sensor-frame tri-axial accelerations, local gyroscopic angular rates, sensor orientation quaternions, and relative inter-segment kinematics. Verify with `tests/test_imu_physics.py`.
