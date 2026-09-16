import sys
import numpy as np

from .components import (
    ExoImuComponent,
    MocapImuComponent,
    MotorComponent,
    ReferenceVideoComponent,
)
from .sync_utils import add_alignment_info, extract_refticks_from_cameras


def parse_modalities_dict(
    names: list[str],
    files: list[str],
    videos: list[str | None],
    datasets: list[str],
    offsets: list[float],
) -> dict[str, dict]:
    """Validates CLI modality lists and returns a structured dictionary keyed by modality name."""
    if (
        len(names) != len(files)
        or len(videos) != len(files)
        or len(datasets) != len(files)
        or len(offsets) != len(files)
    ):
        sys.exit("Error: The number of --name, --video, --dataset, and --offset arguments must match.")

    modalities = dict(
        map(
            lambda x: (x[0], dict(zip(["file", "video", "dataset", "offset"], x[1:]))),
            zip(names, files, videos, datasets, offsets),
        )
    )

    if "motor" not in modalities or ("mocap_imu" not in modalities and "imu" not in modalities):
        sys.exit("Error: 'motor' and 'imu' modalities must be provided.")

    return modalities


def setup_modalities_and_trial(
    modalities: dict[str, dict],
) -> tuple[
    MotorComponent,
    ExoImuComponent | MocapImuComponent,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Initializes Motor and IMU components, extracts trial time bounds,
    and slices the synchronized trial data.

    Returns:
        tuple containing:
            - motor component
            - imu component
            - start_trial_toa
            - end_trial_toa
            - motor_timestamps
            - motor_data
            - imu_timestamps
            - imu_data
    """
    print("Initializing MotorComponent and ExoImuComponent...", flush=True)
    motor = MotorComponent(
        unique_id="motor",
        hdf5_path=modalities["motor"]["file"],
        data_path=modalities["motor"]["dataset"],
        legend_name="Motor",
        offset=modalities["motor"]["offset"],
    )

    if "mocap_imu" in modalities:
        imu = MocapImuComponent(
            unique_id="mocap_imu",
            hdf5_path=modalities["mocap_imu"]["file"],
            data_path=modalities["mocap_imu"]["dataset"],
            legend_name="MoCap IMU",
            offset=modalities["mocap_imu"]["offset"],
        )
    else:
        imu = ExoImuComponent(
            unique_id="imu",
            hdf5_path=modalities["imu"]["file"],
            data_path=modalities["imu"]["dataset"],
            legend_name="IMU",
            offset=modalities["imu"]["offset"],
        )

    # Determine trial start and end TOA
    if "camera" in modalities and modalities["camera"]["video"] not in (None, "-"):
        print("Using Reference Camera for trial boundary extraction...", flush=True)
        cam = ReferenceVideoComponent(
            unique_id="cam",
            video_filepath=modalities["camera"]["video"],
            hdf5_filepath=modalities["camera"]["file"],
            data_path=modalities["camera"]["dataset"],
            legend_name="Camera",
            offset=modalities["camera"]["offset"],
        )
        (
            _,
            _,
            camera_align_info,
            start_trial_toa,
            end_trial_toa,
        ) = extract_refticks_from_cameras([cam])
        add_alignment_info([cam], camera_align_info)
    else:
        print("Using overlapping TOA bounds between Motor and IMU...", flush=True)
        start_trial_toa = max(motor._first_timestamp, imu._first_timestamp)
        end_trial_toa = min(motor._last_timestamp, imu._last_timestamp)

    print(
        f"Trial range: {start_trial_toa:.3f} to {end_trial_toa:.3f} "
        f"({end_trial_toa - start_trial_toa:.2f} seconds)",
        flush=True,
    )

    # Slice data for the trial range using get_frame_for_toa
    imu_idx_start = imu.get_frame_for_toa(start_trial_toa)
    imu_idx_end = imu.get_frame_for_toa(end_trial_toa)
    motor_idx_start = motor.get_frame_for_toa(start_trial_toa)
    motor_idx_end = motor.get_frame_for_toa(end_trial_toa)

    imu_sync = imu.get_sync_info()
    motor_sync = motor.get_sync_info()

    imu_data = imu_sync["data"][imu_idx_start:imu_idx_end]
    imu_timestamps = imu_sync["timestamps"][imu_idx_start:imu_idx_end]

    motor_data = motor_sync["data"][motor_idx_start:motor_idx_end]
    motor_timestamps = motor_sync["timestamps"][motor_idx_start:motor_idx_end]

    return (
        motor,
        imu,
        start_trial_toa,
        end_trial_toa,
        motor_timestamps,
        motor_data,
        imu_timestamps,
        imu_data,
    )
