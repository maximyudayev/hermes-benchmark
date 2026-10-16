"""Phase 2 Stage A: Dual-Stage Camera Calibration - Intrinsic Solver.

Extracts camera intrinsic parameters (matrix K and distortion coefficients D)
from checkerboard or ChArUco calibration image sets or video sequences using OpenCV.
Outputs parameters compatible with Pose2Sim / Aniposelib Calib.toml specifications.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calibrate_camera_checkerboard(
    images_or_pattern: Union[List[Union[str, Path]], str, Path],
    corners_nb: Tuple[int, int] = (9, 6),
    square_size_mm: float = 30.0,
    subpix_criteria: Optional[Tuple[int, int, float]] = None,
) -> Dict[str, Any]:
    """Calibrates a single camera using checkerboard corner detection.

    Args:
        images_or_pattern: Glob pattern or list of filepaths to calibration frames.
        corners_nb: Number of internal corners (cols, rows), e.g. (9, 6).
        square_size_mm: Physical dimension of checkerboard square in millimeters.
        subpix_criteria: Termination criteria for cornerSubPix refinement.

    Returns:
        Dictionary containing:
            - 'matrix': 3x3 camera matrix K
            - 'distortions': [k1, k2, p1, p2, k3]
            - 'image_size': [width, height]
            - 'reproj_error': Root-mean-square reprojection error in pixels
            - 'valid_frames_count': Number of successfully detected frames
    """
    if isinstance(images_or_pattern, (str, Path)):
        image_files = sorted(glob.glob(str(images_or_pattern)))
    else:
        image_files = [str(f) for f in images_or_pattern]

    if not image_files:
        raise FileNotFoundError(f"No calibration images found for pattern/list: {images_or_pattern}")

    # Prepare 3D object points in board reference frame (Z = 0)
    cols, rows = corners_nb
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    img_size = None

    if subpix_criteria is None:
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fpath in image_files:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)

    if not objpoints:
        raise RuntimeError(
            f"Failed to detect checkerboard corners in any of the {len(image_files)} provided images."
        )

    logger.info(f"Successfully detected corners in {len(objpoints)} / {len(image_files)} frames.")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Calculate overall mean reprojection error
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objpoints[i])
    mean_reproj_rmse = np.sqrt(total_error / total_points)

    dist_flat = dist.flatten().tolist()
    # Ensure 5 coefficients [k1, k2, p1, p2, k3]
    if len(dist_flat) < 5:
        dist_flat.extend([0.0] * (5 - len(dist_flat)))

    return {
        "matrix": mtx.tolist(),
        "distortions": dist_flat[:5],
        "image_size": list(img_size),
        "reproj_error": float(mean_reproj_rmse),
        "valid_frames_count": len(objpoints),
    }


def generate_nominal_intrinsics(
    camera_ids: List[str],
    focal_length_mm: float = 4.0,
    sensor_full_res: Tuple[int, int] = (2592, 1944),
    raw_crop_res: Tuple[int, int] = (2560, 1440),
    encoded_res: Tuple[int, int] = (1920, 1080),
    pixel_pitch_um: float = 1.4,
) -> Dict[str, Dict[str, Any]]:
    """Calculates nominal intrinsic camera matrix K based on physical sensor specs.

    Sensor Pipeline:
    1. Full sensor: 2592 x 1944 px (1.4 um pixel pitch -> 3.63 x 2.72 mm).
       Focal length in pixels on sensor: f_sensor = focal_length_mm / (pixel_pitch_um * 1e-3).
    2. Raw center crop: 2560 x 1440 px (aspect ratio crop; pixel scale unchanged).
    3. Encoded video: Scaled to 1920 x 1080 px (scaling factor s = 1920 / 2560 = 0.75).
       Effective focal length: f_video = f_sensor * (W_encoded / W_raw_crop).
       Principal point: c_x = W_encoded / 2.0, c_y = H_encoded / 2.0.

    Returns:
        Mapping of cam_id -> intrinsic parameters dict.
    """
    f_sensor = focal_length_mm / (pixel_pitch_um * 1e-3)
    scale_factor = encoded_res[0] / raw_crop_res[0]
    fx = fy = f_sensor * scale_factor
    cx = encoded_res[0] / 2.0
    cy = encoded_res[1] / 2.0

    k_matrix = [
        [round(fx, 4), 0.0, round(cx, 2)],
        [0.0, round(fy, 4), round(cy, 2)],
        [0.0, 0.0, 1.0],
    ]
    distortions = [0.0, 0.0, 0.0, 0.0, 0.0]

    intrinsics: Dict[str, Dict[str, Any]] = {}
    for cid in camera_ids:
        intrinsics[cid] = {
            "matrix": k_matrix,
            "distortions": distortions,
            "image_size": [encoded_res[0], encoded_res[1]],
            "focal_length_px": round(fx, 2),
            "reproj_error": 0.0,
        }

    logger.info(
        f"Generated nominal intrinsics: fx=fy={fx:.2f} px, cx={cx:.1f}, cy={cy:.1f} "
        f"for cameras: {camera_ids}"
    )
    return intrinsics


def export_intrinsics_toml(
    intrinsics_dict: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Exports intrinsic calibration results into TOML format compatible with Pose2Sim Calib.toml."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Camera Intrinsics generated by Video2Motion intrinsic calibrator\n"]
    for cam_id, data in intrinsics_dict.items():
        lines.append(f"[{cam_id}]")
        lines.append(f"name = \"{cam_id}\"")
        lines.append(f"size = {data['image_size']}")
        lines.append(f"matrix = {data['matrix']}")
        lines.append(f"distortions = {data['distortions']}")
        lines.append(f"# RMS Reprojection Error: {data.get('reproj_error', 0.0):.4f} px\n")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved camera intrinsics to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Stage A: Intrinsic Calibration Solver")
    parser.add_argument("--images", type=str, help="Glob pattern for calibration images (e.g. 'cam01/*.png')")
    parser.add_argument("--cam-id", type=str, default="cam_01", help="Camera identifier")
    parser.add_argument("--corners", nargs=2, type=int, default=[9, 6], help="Checkerboard internal corners (cols rows)")
    parser.add_argument("--square-size", type=float, default=30.0, help="Square dimension in mm (default: 30.0)")
    parser.add_argument("--nominal", action="store_true", help="Generate estimated/nominal intrinsics from sensor specs")
    parser.add_argument("--camera-ids", nargs="+", default=["40478064", "40549960", "40549975", "40549976"], help="Camera IDs for nominal matrix")
    parser.add_argument("--focal-length-mm", type=float, default=4.0, help="Lens focal length in mm (default: 4.0)")
    parser.add_argument("--output-toml", type=str, default="data/calibration/intrinsics/intrinsics.toml")

    args = parser.parse_args()

    if args.nominal:
        intrinsics_data = generate_nominal_intrinsics(
            camera_ids=args.camera_ids,
            focal_length_mm=args.focal_length_mm,
        )
        export_intrinsics_toml(intrinsics_data, args.output_toml)
        return

    if not args.images:
        raise ValueError("Either --images (for checkerboard solving) or --nominal must be provided.")

    results = calibrate_camera_checkerboard(
        images_or_pattern=args.images,
        corners_nb=(args.corners[0], args.corners[1]),
        square_size_mm=args.square_size,
    )

    logger.info(f"Calibration finished. Reprojection RMSE: {results['reproj_error']:.3f} px")
    export_intrinsics_toml({args.cam_id: results}, args.output_toml)


if __name__ == "__main__":
    main()
