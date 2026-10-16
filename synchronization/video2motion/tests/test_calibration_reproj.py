"""Unit tests for Phase 2: Dual-Stage Camera Calibration Reprojection Accuracy.

Pass Criteria:
- Mean Reprojection Error < 2.0 pixels.
- Accurate recovery of camera rotation and translation vectors.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from importlib import import_module
scene_calibrator = import_module("03_scene_calibrator")


def test_pnp_reprojection_error_within_threshold():
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not installed in active environment.")

    # 1. Define synthetic ground truth 3D landmarks (>= 6 non-coplanar points)
    object_points_3d = np.array([
        [0.00, 0.00, 0.00],
        [1.20, 0.00, 0.00],
        [0.00, 1.80, 0.00],
        [0.00, 0.00, 2.10],
        [1.50, 1.80, 0.45],
        [2.40, 0.90, 0.00],
        [0.80, 1.20, 1.50],
    ], dtype=np.float64)

    # 2. Known camera intrinsics (1920x1080 camera, fx=fy=1200, cx=960, cy=540)
    k_true = np.array([
        [1200.0, 0.0, 960.0],
        [0.0, 1200.0, 540.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_true = np.array([0.02, -0.01, 0.001, 0.0005, 0.0], dtype=np.float64)

    # 3. Ground truth camera pose (looking at origin from [2.0, -3.0, 1.8])
    rvec_true = np.array([1.2, 0.4, -0.3], dtype=np.float64)
    tvec_true = np.array([0.2, -0.5, 3.5], dtype=np.float64)

    # 4. Forward project to obtain synthetic 2D points with sub-pixel noise (+- 0.3 px)
    proj_pts_2d, _ = cv2.projectPoints(object_points_3d, rvec_true, tvec_true, k_true, dist_true)
    proj_pts_2d = proj_pts_2d.reshape(-1, 2)

    np.random.seed(42)
    noisy_pts_2d = proj_pts_2d + np.random.normal(0.0, 0.3, proj_pts_2d.shape)

    # 5. Solve PnP
    rvec_est, tvec_est, reproj_rmse = scene_calibrator.solve_camera_extrinsics_pnp(
        object_points_3d=object_points_3d,
        image_points_2d=noisy_pts_2d,
        camera_matrix_k=k_true,
        dist_coeffs=dist_true,
    )

    # Assertion: Mean Reprojection Error < 2.0 pixels
    assert reproj_rmse < 2.0, f"Reprojection error {reproj_rmse:.3f} px exceeds 2.0 px threshold."

    # Rotation and translation should be accurately recovered
    np.testing.assert_allclose(rvec_est.flatten(), rvec_true, atol=0.05)
    np.testing.assert_allclose(tvec_est.flatten(), tvec_true, atol=0.08)
