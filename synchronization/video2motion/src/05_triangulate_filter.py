"""Phase 4: Robust Triangulation & Kinematic Filtering.

Orchestrates multi-view 2D proposal triangulation into continuous 3D coordinate trajectories,
filters tracking jitter using zero-lag Butterworth filtering, and augments occluded markers.
Invokes Pose2Sim modules or provides standalone filtering fallback.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from Pose2Sim import Pose2Sim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pose2sim_pipeline(
    config_path: Union[str, Path] = "configs/Config.toml",
    do_triangulation: bool = True,
    do_filtering: bool = True,
    do_augmentation: bool = True,
) -> None:
    """Executes the standard Pose2Sim triangulation, filtering, and marker augmentation steps."""
    config_path = str(config_path)

    if do_triangulation:
        logger.info(f"Running Pose2Sim.triangulation('{config_path}')...")
        Pose2Sim.triangulation(config_path)

    if do_filtering:
        logger.info(f"Running Pose2Sim.filtering('{config_path}')...")
        Pose2Sim.filtering(config_path)

    if do_augmentation:
        logger.info(f"Running Pose2Sim.markerAugmentation('{config_path}')...")
        Pose2Sim.markerAugmentation(config_path)

    logger.info("Pose2Sim processing stages complete.")


def apply_butterworth_filter(
    data: np.ndarray,
    sampling_rate: float = 30.0,
    cutoff_freq: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Applies a zero-lag low-pass Butterworth filter across axis 0."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    if normal_cutoff >= 1.0:
        raise ValueError(
            f"Cutoff frequency ({cutoff_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz)."
        )

    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Filter with nan handling (linear interpolation over internal nans)
    filtered = np.copy(data)
    if filtered.ndim == 1:
        filtered = filtered[:, np.newaxis]

    for col in range(filtered.shape[1]):
        col_vals = filtered[:, col]
        nans = np.isnan(col_vals)
        if np.all(nans):
            continue
        if np.any(nans):
            # Interpolate nans before filtering
            valid_idx = np.where(~nans)[0]
            col_vals[nans] = np.interp(np.where(nans)[0], valid_idx, col_vals[valid_idx])

        col_filtered = filtfilt(b, a, col_vals)
        filtered[:, col] = col_filtered

    return filtered.squeeze()


def write_trc_file(
    output_path: Union[str, Path],
    marker_names: list[str],
    timestamps: np.ndarray,
    marker_data_3d: np.ndarray,  # Shape (N_frames, N_markers, 3) in millimeters
    data_rate: float = 30.0,
    camera_rate: float = 30.0,
    units: str = "mm",
) -> None:
    """Writes 3D marker coordinates into standard Motion Analysis TRC (.trc) file format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    num_frames = len(timestamps)
    num_markers = len(marker_names)

    lines = [
        "PathFileType\t4\t(X/Y/Z)\toutput.trc",
        f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.1f}\t{camera_rate:.1f}\t{num_frames}\t{num_markers}\t{units}\t{data_rate:.1f}\t1\t{num_frames}",
    ]

    # Marker header line 1: Frame# Time Marker1 Marker2 ...
    h1 = ["Frame#", "Time"]
    for m in marker_names:
        h1.extend([m, "", ""])
    lines.append("\t".join(h1))

    # Marker header line 2: X1 Y1 Z1 X2 Y2 Z2 ...
    h2 = ["", ""]
    for i in range(1, num_markers + 1):
        h2.extend([f"X{i}", f"Y{i}", f"Z{i}"])
    lines.append("\t".join(h2))
    lines.append("")  # Blank separator line

    # Data rows
    for f_idx in range(num_frames):
        row = [str(f_idx + 1), f"{timestamps[f_idx]:.5f}"]
        for m_idx in range(num_markers):
            vals = marker_data_3d[f_idx, m_idx]
            row.extend([f"{vals[0]:.3f}", f"{vals[1]:.3f}", f"{vals[2]:.3f}"])
        lines.append("\t".join(row))

    out_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"TRC file successfully written to {out_file} ({num_frames} frames, {num_markers} markers)")


def parse_trc_file(trc_path: Union[str, Path]) -> Tuple[list[str], np.ndarray, np.ndarray]:
    """Reads a .trc file into marker names, timestamps, and 3D coordinate array (N, M, 3)."""
    trc_path = Path(trc_path)
    with open(trc_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\r\n") for line in f if line.strip()]

    # Line 3 contains marker names: Frame# \t Time \t Marker1 \t \t \t Marker2 ...
    header_tokens = raw_lines[3].split("\t")
    marker_names = [name.strip() for name in header_tokens[2:] if name.strip()]

    data_lines = raw_lines[5:]
    num_frames = len(data_lines)
    num_markers = len(marker_names)

    timestamps = np.zeros(num_frames)
    coords = np.zeros((num_frames, num_markers, 3))

    for i, line in enumerate(data_lines):
        tokens = line.split("\t")
        timestamps[i] = float(tokens[1])
        c_vals = [float(v) for v in tokens[2:] if v.strip()]
        for m in range(num_markers):
            coords[i, m, 0] = c_vals[m * 3]
            coords[i, m, 1] = c_vals[m * 3 + 1]
            coords[i, m, 2] = c_vals[m * 3 + 2]

    return marker_names, timestamps, coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Triangulation and Kinematic Filtering")
    parser.add_argument("--config", type=str, default="configs/Config.toml", help="Path to Pose2Sim Config.toml")
    parser.add_argument("--skip-aug", action="store_true", help="Skip marker augmentation step")

    args = parser.parse_args()
    run_pose2sim_pipeline(
        config_path=args.config,
        do_triangulation=True,
        do_filtering=True,
        do_augmentation=not args.skip_aug,
    )


if __name__ == "__main__":
    main()
