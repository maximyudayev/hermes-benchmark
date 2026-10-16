"""Unit tests for Phase 1: Timestamp Harmonization & Grid Alignment.

Validation Criteria:
- Absolute time error < 1e-4 seconds between consecutive master frames and mapped timestamps.
- Correct detection of missing frames and sentinel placement (cam_valid = False, src_frame = -1).
- Correct temporal overlap window derivation.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from importlib import import_module
sync_aligner = import_module("01_sync_aligner")


def test_master_timeline_grid_integrity():
    fps = 30.0
    dt = 1.0 / fps
    n_frames = 100

    # Synthetic camera 1: exact uniform clock with 1 dropped frame at index 40
    ts_cam1 = np.arange(n_frames) * dt + 10.0
    idx_cam1 = np.arange(n_frames)
    # Drop frame 40
    ts_cam1 = np.delete(ts_cam1, 40)
    idx_cam1 = np.delete(idx_cam1, 40)

    # Synthetic camera 2: slight jitter within tolerance
    jitter = np.random.uniform(-0.1 * dt, 0.1 * dt, n_frames)
    ts_cam2 = np.arange(n_frames) * dt + 10.0 + jitter
    idx_cam2 = np.arange(n_frames)

    manifest = sync_aligner.build_master_timeline(
        camera_sources={
            "cam_01": (idx_cam1, ts_cam1),
            "cam_02": (idx_cam2, ts_cam2),
        },
        nominal_fps=fps,
        reference_camera="cam_01",
    )

    assert not manifest.empty
    assert "master_frame" in manifest.columns
    assert "cam_01_valid" in manifest.columns
    assert "cam_02_valid" in manifest.columns

    # 1. Delta between consecutive master frames should be dt within 1e-6
    master_ts = manifest["timestamp"].values
    deltas = np.diff(master_ts)
    np.testing.assert_allclose(deltas, dt, atol=1e-6)

    # 2. Check dropped frame detection on cam_01
    missing_cam1 = manifest[~manifest["cam_01_valid"]]
    assert len(missing_cam1) >= 1
    assert -1 in missing_cam1["cam_01_src_frame"].values

    # 3. Time error on valid matched frames must be strictly within tolerance (dt / 2)
    valid_cam2 = manifest[manifest["cam_02_valid"]]
    assert (valid_cam2["cam_02_time_error"] <= (dt / 2.0)).all()
    # Check absolute time error < 1e-4 s for perfectly aligned frames
    valid_cam1 = manifest[manifest["cam_01_valid"]]
    assert (valid_cam1["cam_01_time_error"] < 1e-4).all()


def test_missing_frame_json_emission(tmp_path):
    pose_detector = import_module("04_pose_detector")
    json_path = tmp_path / "dummy_000040_keypoints.json"

    pose_detector.write_missing_frame_json(json_path, num_keypoints=26)
    assert json_path.is_file()

    import json
    with open(json_path) as f:
        data = json.load(f)

    pts = data["people"][0]["pose_keypoints_2d"]
    assert len(pts) == 26 * 3
    # Check all confidences are 0.0
    confidences = pts[2::3]
    assert all(c == 0.0 for c in confidences)


def test_load_hermes_session_schema(tmp_path):
    import h5py
    h5_path = tmp_path / "cameras.hdf5"
    mkv_1 = tmp_path / "cameras_40478064.mkv"
    mkv_2 = tmp_path / "cameras_40549960.mkv"
    mkv_1.touch()
    mkv_2.touch()

    # Create synthetic HERMES cameras.hdf5
    with h5py.File(h5_path, "w") as f:
        g1 = f.create_group("cameras/40478064")
        g1.create_dataset("toa_s", data=np.array([[100.0], [100.033], [100.066]]))
        g1.create_dataset("frame_index", data=np.array([[0], [1], [2]]))

        g2 = f.create_group("cameras/40549960")
        g2.create_dataset("toa_s", data=np.array([[100.0], [100.033], [100.066]]))
        g2.create_dataset("frame_index", data=np.array([[0], [1], [2]]))

    cam_sources, vid_paths = sync_aligner.load_hermes_session(tmp_path)
    assert "40478064" in cam_sources
    assert "40549960" in cam_sources
    assert "40478064" in vid_paths
    assert "40549960" in vid_paths
    assert len(cam_sources["40478064"][1]) == 3

