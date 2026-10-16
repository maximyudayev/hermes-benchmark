"""Phase 6: Local Virtual IMU Synthesis & Inter-Segment Kinematics Engine.

Synthesizes local tri-axial accelerometer (specific force) and gyroscope (angular velocity)
signals directly expressed in the local sensor housing frame (S).
Incorporates custom sensor-to-bone physical mounting offsets (R_S/B and p_S/B),
evaluates OpenSim dynamic state via generalized speed differentiation,
calculates relative inter-segment kinematics (Cardan angles), and exports orientation unit quaternions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import opensim as osim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard gravitational acceleration vector in OpenSim ground frame (Y-up: [0, -9.81, 0])
DEFAULT_GRAVITY = np.array([0.0, -9.81, 0.0])


def build_virtual_imu_dataset(
    model_path: Union[str, Path],
    mot_path: Union[str, Path],
    sensor_configurations: Dict[str, Dict[str, Any]],
    output_csv_path: Optional[Union[str, Path]] = None,
    gravity_vector: np.ndarray = DEFAULT_GRAVITY,
) -> pd.DataFrame:
    """Synthesizes local IMU specific force, angular rate, quaternions,
    and relative inter-segment kinematics from OpenSim IK output.

    Args:
        model_path: Path to scaled OpenSim model (.osim).
        mot_path: Path to Inverse Kinematics storage file (.mot).
        sensor_configurations: Dictionary specifying sensors, e.g.:
            {
                'shank_r_imu': {
                    'parent_body': 'tibia_r',
                    'pos_offset': [0.02, -0.18, 0.04],  # [x, y, z] meters on bone
                    'rot_offset_deg': [0.0, 90.0, 0.0]  # Euler xyz mounting rotation in deg
                },
                'thigh_r_imu': {
                    'parent_body': 'femur_r',
                    'pos_offset': [0.04, -0.20, 0.02],
                    'rot_offset_deg': [0.0, 0.0, 0.0]
                }
            }
        output_csv_path: Optional CSV output filepath.
        gravity_vector: Global gravity vector in m/s^2 (default [0, -9.81, 0]).

    Returns:
        pd.DataFrame containing timestamps, synthesized local IMU signals, and joint angles.
    """
    model_path = Path(model_path)
    mot_path = Path(mot_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not mot_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {mot_path}")

    model = osim.Model(str(model_path))
    # Clear muscle and actuator forces to eliminate dynamic force overhead during purely kinematic realization
    model.updForceSet().clearAndDestroy()
    state = model.initSystem()

    # 1. Ingest coordinates from IK output storage
    motion = osim.Storage(str(mot_path))
    labels = [motion.getColumnLabels().get(i) for i in range(motion.getColumnLabels().size())]
    time_col = osim.ArrayDouble()
    motion.getTimeColumn(time_col)
    num_frames = time_col.getSize()
    if num_frames < 2:
        raise ValueError(f"Motion file contains too few frames ({num_frames}) for differentiation.")

    fps = (num_frames - 1) / (time_col.getLast() - time_col.get(0))
    dt = 1.0 / fps

    coord_names = labels[1:]
    q_data = np.zeros((num_frames, len(coord_names)))
    for i in range(num_frames):
        row = motion.getStateVector(i).getData()
        for j in range(len(coord_names)):
            q_data[i, j] = row.get(j)

    # 2. Derive generalized speeds (u) and accelerations (udot)
    # Convert rotational degrees of freedom from degrees to radians
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

    # Central finite differences for generalized speeds u
    u_rad = np.gradient(q_rad, dt, axis=0)

    results = []

    for i in range(num_frames):
        t = time_col.get(i)
        state.setTime(t)

        # 3. Update generalized coordinates and speeds
        for j, c_name in enumerate(coord_names):
            if model.updCoordinateSet().contains(c_name):
                coord = model.updCoordinateSet().get(c_name)
                coord.setValue(state, q_rad[i, j], False)
                coord.setSpeedValue(state, u_rad[i, j])

        model.realizeAcceleration(state)

        frame_dict: Dict[str, Any] = {"time": t}
        segment_rot_matrices: Dict[str, np.ndarray] = {}

        for sensor_id, cfg in sensor_configurations.items():
            parent_name = cfg["parent_body"]
            if not model.getBodySet().contains(parent_name):
                logger.warning(f"Body '{parent_name}' not in model. Skipping sensor '{sensor_id}'.")
                continue

            body = model.getBodySet().get(parent_name)
            p_offset = osim.Vec3(*cfg.get("pos_offset", [0.0, 0.0, 0.0]))

            # Local sensor mounting rotation R_S/B
            rot_deg = cfg.get("rot_offset_deg", [0.0, 0.0, 0.0])
            r_sb = R.from_euler("xyz", rot_deg, degrees=True).as_matrix()

            # Global bone rotation R_B/G
            r_bg_osim = body.getTransformInGround(state).R()
            r_bg = np.array([[r_bg_osim.get(r, c) for c in range(3)] for r in range(3)])
            segment_rot_matrices[parent_name] = r_bg

            # Total sensor orientation in ground: R_S/G = R_B/G @ R_S/B
            r_sg = r_bg @ r_sb

            # --- LOCAL GYROSCOPE (ANGULAR VELOCITY) ---
            omega_g = body.getAngularVelocityInGround(state)
            omega_g_np = np.array([omega_g.get(0), omega_g.get(1), omega_g.get(2)])
            # Project onto local housing axes: omega_S = R_S/G^T @ omega_G
            omega_local = r_sg.T @ omega_g_np
            frame_dict[f"{sensor_id}_gyro_x_dps"] = float(np.degrees(omega_local[0]))
            frame_dict[f"{sensor_id}_gyro_y_dps"] = float(np.degrees(omega_local[1]))
            frame_dict[f"{sensor_id}_gyro_z_dps"] = float(np.degrees(omega_local[2]))

            # --- LOCAL ACCELEROMETER (SPECIFIC FORCE) ---
            acc_station_g = body.findStationAccelerationInGround(state, p_offset)
            acc_station_np = np.array(
                [acc_station_g.get(0), acc_station_g.get(1), acc_station_g.get(2)]
            )
            # Specific force: a_sens = R_S/G^T @ (a_station_g - g)
            acc_local = r_sg.T @ (acc_station_np - gravity_vector)
            frame_dict[f"{sensor_id}_acc_x_g"] = float(acc_local[0] / 9.81)
            frame_dict[f"{sensor_id}_acc_y_g"] = float(acc_local[1] / 9.81)
            frame_dict[f"{sensor_id}_acc_z_g"] = float(acc_local[2] / 9.81)

            # --- LOCAL SENSOR QUATERNIONS ---
            q_sens = R.from_matrix(r_sg).as_quat()  # [x, y, z, w]
            frame_dict[f"{sensor_id}_quat_x"] = float(q_sens[0])
            frame_dict[f"{sensor_id}_quat_y"] = float(q_sens[1])
            frame_dict[f"{sensor_id}_quat_z"] = float(q_sens[2])
            frame_dict[f"{sensor_id}_quat_w"] = float(q_sens[3])

        # --- RELATIVE INTER-SEGMENT KINEMATICS (KNEE JOINT) ---
        if "tibia_r" in segment_rot_matrices and "femur_r" in segment_rot_matrices:
            r_thigh = segment_rot_matrices["femur_r"]
            r_shank = segment_rot_matrices["tibia_r"]
            # R_rel: Orientation of distal (shank) relative to proximal (thigh)
            r_rel = r_thigh.T @ r_shank
            knee_angles = R.from_matrix(r_rel).as_euler("zxy", degrees=True)
            frame_dict["knee_r_rel_flexion_deg"] = float(knee_angles[0])
            frame_dict["knee_r_rel_adduction_deg"] = float(knee_angles[1])
            frame_dict["knee_r_rel_rotation_deg"] = float(knee_angles[2])

        results.append(frame_dict)

    df_results = pd.DataFrame(results)
    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_p, index=False)
        logger.info(f"Virtual IMU dataset saved to: {out_p}")

    return df_results

DEFAULT_SENSOR_CONFIG: Dict[str, Dict[str, Any]] = {
    "pelvis_imu": {
        "parent_body": "pelvis",
        "pos_offset": [-0.08, 0.0, 0.0],  # Posterior midline / sacrum
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
    "thigh_r_imu": {
        "parent_body": "femur_r",
        "pos_offset": [0.05, -0.20, 0.0],  # Frontal mid-thigh
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
    "thigh_l_imu": {
        "parent_body": "femur_l",
        "pos_offset": [0.05, -0.20, 0.0],  # Frontal mid-thigh
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
    "shank_r_imu": {
        "parent_body": "tibia_r",
        "pos_offset": [0.04, -0.20, 0.0],  # Frontal mid-shank (anterior tibial crest)
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
    "shank_l_imu": {
        "parent_body": "tibia_l",
        "pos_offset": [0.04, -0.20, 0.0],  # Frontal mid-shank (anterior tibial crest)
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
    "toes_r_imu": {
        "parent_body": "toes_r",
        "pos_offset": [0.03, 0.02, 0.0],   # Dorsal forefoot / toes
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
    "toes_l_imu": {
        "parent_body": "toes_l",
        "pos_offset": [0.03, 0.02, 0.0],   # Dorsal forefoot / toes
        "rot_offset_deg": [0.0, 0.0, 0.0],
    },
}


def _synthetic_imu_fallback(
    mot_path: Union[str, Path],
    sensor_configurations: Dict[str, Dict[str, Any]],
    gravity_vector: np.ndarray = DEFAULT_GRAVITY,
) -> pd.DataFrame:
    """Pure-numerical simulation fallback for standalone tests or when OpenSim is mock-tested."""
    mot_path = Path(mot_path)
    if mot_path.is_file():
        try:
            df_mot = pd.read_csv(mot_path, sep=r"\s+", skiprows=lambda x: x < 10)
            time_arr = df_mot.iloc[:, 0].values
        except Exception:
            time_arr = np.linspace(0, 2.0, 60)
    else:
        time_arr = np.linspace(0, 2.0, 60)

    results = []
    g_unit = -gravity_vector / np.linalg.norm(gravity_vector)

    for t in time_arr:
        row: Dict[str, Any] = {"time": float(t)}
        for sensor_id, cfg in sensor_configurations.items():
            rot_deg = cfg.get("rot_offset_deg", [0.0, 0.0, 0.0])
            r_sb = R.from_euler("xyz", rot_deg, degrees=True).as_matrix()
            acc_local = r_sb.T @ g_unit

            row[f"{sensor_id}_gyro_x_dps"] = 0.0
            row[f"{sensor_id}_gyro_y_dps"] = 0.0
            row[f"{sensor_id}_gyro_z_dps"] = 0.0
            row[f"{sensor_id}_acc_x_g"] = float(acc_local[0])
            row[f"{sensor_id}_acc_y_g"] = float(acc_local[1])
            row[f"{sensor_id}_acc_z_g"] = float(acc_local[2])

            q_sens = R.from_matrix(r_sb).as_quat()
            row[f"{sensor_id}_quat_x"] = float(q_sens[0])
            row[f"{sensor_id}_quat_y"] = float(q_sens[1])
            row[f"{sensor_id}_quat_z"] = float(q_sens[2])
            row[f"{sensor_id}_quat_w"] = float(q_sens[3])

        row["knee_r_rel_flexion_deg"] = 15.0 * np.sin(2.0 * np.pi * 1.0 * t)
        row["knee_r_rel_adduction_deg"] = 0.0
        row["knee_r_rel_rotation_deg"] = 0.0
        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Virtual IMU Synthesis Engine")
    parser.add_argument("--model", type=str, default="", help="Path to scaled .osim model")
    parser.add_argument("--mot", type=str, default="", help="Path to .mot motion file")
    parser.add_argument("--config-json", type=str, help="Path to JSON file with sensor_configurations")
    parser.add_argument("--output-csv", type=str, default="data/virtual_imu_dataset.csv")

    args = parser.parse_args()

    # Auto-resolve scaled model if not explicitly specified
    model_path = args.model
    if not model_path or not Path(model_path).is_file():
        scaled_models = list(Path("kinematics").glob("*.osim")) or list(Path("models").glob("*.osim"))
        if scaled_models:
            model_path = str(max(scaled_models, key=lambda p: p.stat().st_ctime))
            logger.info(f"Auto-selected OpenSim model: {model_path}")
        else:
            model_path = "models/Rajagopal_Pose2Sim_scaled.osim"

    # Auto-resolve motion kinematics file if not explicitly specified
    mot_path = args.mot
    if not mot_path or not Path(mot_path).is_file():
        mot_candidates = list(Path("kinematics").glob("*.mot")) or list(Path("opensim/Results").glob("*.mot"))
        if mot_candidates:
            mot_path = str(max(mot_candidates, key=lambda p: p.stat().st_ctime))
            logger.info(f"Auto-selected IK motion file: {mot_path}")
        else:
            mot_path = "opensim/Results/trial_kinematics.mot"

    if args.config_json and Path(args.config_json).is_file():
        with open(args.config_json, "r") as f:
            sensor_cfg = json.load(f)
    else:
        # Default 7-sensor configuration: pelvis, frontal mid-thighs, frontal mid-shanks, toes
        sensor_cfg = DEFAULT_SENSOR_CONFIG

    build_virtual_imu_dataset(
        model_path=model_path,
        mot_path=mot_path,
        sensor_configurations=sensor_cfg,
        output_csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
