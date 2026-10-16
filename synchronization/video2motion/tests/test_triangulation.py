"""Unit tests for Phase 4: Triangulation Rigidity & Kinematic Filtering.

Pass Criteria:
- Inter-joint segment length standard deviation < 1.5 mm across trial (rigid bone segment check).
- Zero-lag Butterworth filtering suppresses high-frequency tracking jitter.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from importlib import import_module
triangulate_filter = import_module("05_triangulate_filter")


def test_bone_length_rigidity_post_filtering():
    """Verifies that distance between anatomical pairs (e.g., knee to ankle) maintains rigid body consistency (< 1.5 mm standard deviation)."""
    np.random.seed(42)
    n_frames = 150
    fps = 30.0
    t = np.arange(n_frames) / fps

    # Ground truth: Shank (Tibia) rigid length = 380.0 mm
    tibia_length_true = 380.0

    # Moving knee position (sagittal plane walking trajectory)
    knee_x = 500.0 * np.sin(2 * np.pi * 0.8 * t)
    knee_y = 100.0 * np.cos(2 * np.pi * 0.8 * t) + 450.0
    knee_z = np.zeros(n_frames)

    # Ankle rotating around knee (flexion/extension)
    angle = np.radians(20.0 + 30.0 * np.sin(2 * np.pi * 0.8 * t))
    ankle_x = knee_x + tibia_length_true * np.sin(angle)
    ankle_y = knee_y - tibia_length_true * np.cos(angle)
    ankle_z = np.zeros(n_frames)

    knee_clean = np.column_stack([knee_x, knee_y, knee_z])
    ankle_clean = np.column_stack([ankle_x, ankle_y, ankle_z])

    # Add realistic marker tracking jitter (+- 1.5 mm Gaussian noise)
    jitter_knee = np.random.normal(0.0, 1.5, knee_clean.shape)
    jitter_ankle = np.random.normal(0.0, 1.5, ankle_clean.shape)

    knee_noisy = knee_clean + jitter_knee
    ankle_noisy = ankle_clean + jitter_ankle

    # Check raw noisy length standard deviation exceeds threshold
    raw_lengths = np.linalg.norm(knee_noisy - ankle_noisy, axis=1)
    raw_std = np.std(raw_lengths)
    assert raw_std > 1.5, "Raw noisy test input should exceed 1.5 mm"

    # Apply 4th order 6 Hz Butterworth filter
    knee_filtered = triangulate_filter.apply_butterworth_filter(
        knee_noisy, sampling_rate=fps, cutoff_freq=6.0, order=4
    )
    ankle_filtered = triangulate_filter.apply_butterworth_filter(
        ankle_noisy, sampling_rate=fps, cutoff_freq=6.0, order=4
    )

    # Compute filtered segment lengths
    filtered_lengths = np.linalg.norm(knee_filtered - ankle_filtered, axis=1)
    # Exclude boundary filter transient (first and last 10 frames)
    interior_lengths = filtered_lengths[10:-10]
    filtered_std = float(np.std(interior_lengths))

    # Pass Criteria: Segment length standard deviation < 1.5 mm
    assert filtered_std < 1.5, (
        f"Filtered segment length standard deviation ({filtered_std:.3f} mm) exceeds 1.5 mm limit."
    )


def test_trc_file_roundtrip(tmp_path):
    trc_path = tmp_path / "test_motion.trc"
    marker_names = ["RKnee", "RAnkle"]
    n_frames = 50
    ts = np.linspace(0.0, 1.633, n_frames)
    coords = np.random.uniform(100.0, 500.0, (n_frames, 2, 3))

    triangulate_filter.write_trc_file(
        output_path=trc_path,
        marker_names=marker_names,
        timestamps=ts,
        marker_data_3d=coords,
    )

    assert trc_path.is_file()
    names_read, ts_read, coords_read = triangulate_filter.parse_trc_file(trc_path)

    assert names_read == marker_names
    np.testing.assert_allclose(ts_read, ts, atol=1e-4)
    np.testing.assert_allclose(coords_read, coords, atol=1e-3)
