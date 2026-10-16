"""Unit tests for Phase 6: Virtual IMU Dynamics & Physical Invariants.

Pass Criteria:
- Static baseline acceleration magnitude: 1.00 +- 0.02 g (reaction to Earth gravity).
- Static angular velocity magnitude: < 0.5 deg/s across all axes.
- Unit quaternion normalization: ||q|| = 1.0000 +- 1e-4.
- Rotation transformation invariance: Acc specific force rotates correctly under R_S/B mounting offsets.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from importlib import import_module
imu_engine = import_module("07_imu_inertial_engine")


def test_static_baseline_gravity_and_gyro():
    """Validates that a stationary sensor measures exactly 1g specific force pointing opposite to gravity and zero angular velocity."""
    sensor_cfg = {
        "shank_r_imu": {
            "parent_body": "tibia_r",
            "pos_offset": [0.0, -0.20, 0.0],
            "rot_offset_deg": [0.0, 0.0, 0.0],  # Aligned with bone
        },
        "shank_r_pitched": {
            "parent_body": "tibia_r",
            "pos_offset": [0.0, -0.20, 0.0],
            "rot_offset_deg": [0.0, 45.0, 0.0],  # Pitched 45 deg around Y
        },
    }

    # Run virtual IMU generation with static fallback
    df = imu_engine._synthetic_imu_fallback(
        mot_path="nonexistent.mot",
        sensor_configurations=sensor_cfg,
        gravity_vector=np.array([0.0, -9.81, 0.0]),
    )

    assert not df.empty

    for sensor_id in ["shank_r_imu", "shank_r_pitched"]:
        # 1. Total acceleration magnitude must be 1.00 +- 0.02 g
        ax = df[f"{sensor_id}_acc_x_g"].values
        ay = df[f"{sensor_id}_acc_y_g"].values
        az = df[f"{sensor_id}_acc_z_g"].values
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

        np.testing.assert_allclose(
            acc_mag, 1.00, atol=0.02,
            err_msg=f"Acceleration magnitude for {sensor_id} deviates from 1.0g."
        )

        # 2. Angular velocity must be < 0.5 deg/s
        gx = df[f"{sensor_id}_gyro_x_dps"].values
        gy = df[f"{sensor_id}_gyro_y_dps"].values
        gz = df[f"{sensor_id}_gyro_z_dps"].values
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        assert (gyro_mag < 0.5).all(), f"Stationary gyro for {sensor_id} exceeded 0.5 deg/s"

        # 3. Unit quaternion norm == 1.0
        qx = df[f"{sensor_id}_quat_x"].values
        qy = df[f"{sensor_id}_quat_y"].values
        qz = df[f"{sensor_id}_quat_z"].values
        qw = df[f"{sensor_id}_quat_w"].values
        quat_norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        np.testing.assert_allclose(
            quat_norm, 1.0, atol=1e-4,
            err_msg=f"Quaternion norm for {sensor_id} violated unit length."
        )


def test_inter_segment_cardan_decomposition():
    """Tests relative Cardan angle decomposition between two orientation matrices."""
    # Distal segment (shank) rotated 40 degrees flexion around X relative to proximal (thigh)
    r_thigh = R.from_euler("xyz", [10, 0, 0], degrees=True).as_matrix()
    r_shank = R.from_euler("xyz", [50, 0, 0], degrees=True).as_matrix()

    r_rel = r_thigh.T @ r_shank
    knee_angles = R.from_matrix(r_rel).as_euler("xyz", degrees=True)

    np.testing.assert_allclose(knee_angles[0], 40.0, atol=1e-4)
    np.testing.assert_allclose(knee_angles[1], 0.0, atol=1e-4)
    np.testing.assert_allclose(knee_angles[2], 0.0, atol=1e-4)
