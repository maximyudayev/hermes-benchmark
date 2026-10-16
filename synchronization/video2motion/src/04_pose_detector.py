"""Phase 3: 2D Multi-View Pose Estimation with Missing-Frame Guarding.

Runs RTMPose / HALPE_26 keypoint inference across synchronized multi-view frames.
Directly reads from raw video feeds (.mkv / .mp4) or time-gridded image directories.
Guards against frame-indexing desynchronization by emitting OpenPose-compliant dummy
JSON records (26 keypoints with confidence 0.0) whenever a camera dropped a frame.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 26


def write_missing_frame_json(output_path: Union[str, Path], num_keypoints: int = NUM_KEYPOINTS) -> None:
    """Writes empty detection JSON to prevent frame-indexing desynchronization in Pose2Sim."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dummy_keypoints = [0.0, 0.0, 0.0] * num_keypoints
    payload = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": dummy_keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
            }
        ],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_valid_frame_json(
    output_path: Union[str, Path],
    keypoints_xy: np.ndarray,
    confidences: np.ndarray,
) -> None:
    """Writes detected 2D landmarks to an OpenPose-compliant JSON record."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_pts = len(keypoints_xy)
    flat_record = []
    for i in range(n_pts):
        flat_record.append(float(keypoints_xy[i, 0]))
        flat_record.append(float(keypoints_xy[i, 1]))
        flat_record.append(float(confidences[i]))

    payload = {
        "version": 1.3,
        "people": [
            {
                "person_id": [0],
                "pose_keypoints_2d": flat_record,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
            }
        ],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def initialize_rtmpose_tracker(
    mode: str = "balanced",  # 'lightweight', 'balanced', or 'performance'
    device: str = "cpu",
):
    """Initializes RTMLib BodyWithFeet pose detector if installed."""
    try:
        from rtmlib import BodyWithFeet
        return BodyWithFeet(mode=mode, to_openpose=True, device=device)
    except ImportError:
        logger.warning(
            "RTMLib is not installed in active environment. Inference in stub mode."
        )
        return None


def process_synchronized_stream(
    manifest_csv: Union[str, Path],
    cam_id: str,
    video_path: Optional[Union[str, Path]] = None,
    frames_dir: Optional[Union[str, Path]] = None,
    output_pose_dir: Union[str, Path] = "pose-2d",
    tracker: Any = None,
    device: str = "cpu",
) -> None:
    """Processes frames for a single camera directly from video or frames folder.

    For each master frame:
    - If camera valid: seeks to src_frame, runs RTMPose HALPE_26, writes OpenPose JSON.
    - If camera dropped (cam_valid == False): writes empty dummy JSON.
    """
    manifest_df = pd.read_csv(manifest_csv)
    out_dir = Path(output_pose_dir) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_col = f"{cam_id}_valid"
    src_col = f"{cam_id}_src_frame"

    if valid_col not in manifest_df.columns:
        raise KeyError(f"Column '{valid_col}' not found in manifest: {manifest_csv}")

    if tracker is None:
        tracker = initialize_rtmpose_tracker(device=device)

    cap = None
    if video_path and Path(video_path).is_file() and cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            cap = None

    logger.info(
        f"Processing 2D pose detection for {cam_id} ({len(manifest_df)} master frames)..."
    )

    missing_count = 0
    valid_count = 0
    last_src_idx = -2

    for _, row in manifest_df.iterrows():
        master_idx = int(row["master_frame"])
        is_valid = bool(row[valid_col])
        src_idx = int(row[src_col]) if src_col in row else -1
        json_name = f"{cam_id}_{master_idx:06d}_keypoints.json"
        json_path = out_dir / json_name

        if not is_valid or src_idx < 0:
            write_missing_frame_json(json_path)
            missing_count += 1
            continue

        img = None
        # Mode 1: Direct seek from video file (fast, no disk footprint)
        if cap is not None:
            if src_idx != last_src_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ret, frame = cap.read()
            last_src_idx = src_idx
            if ret and frame is not None:
                img = frame

        # Mode 2: Read from pre-extracted image directory
        elif frames_dir:
            f_dir = Path(frames_dir)
            f_path = f_dir / f"{cam_id}_{master_idx:06d}.png"
            if not f_path.is_file():
                f_path = f_dir / f"{cam_id}_{master_idx:06d}.jpg"
            if f_path.is_file() and cv2 is not None:
                img = cv2.imread(str(f_path))

        if img is None or tracker is None:
            write_missing_frame_json(json_path)
            missing_count += 1
            continue

        # Run RTMPose inference
        keypoints, scores = tracker(img)
        if len(keypoints) > 0:
            write_valid_frame_json(json_path, keypoints[0], scores[0])
            valid_count += 1
        else:
            write_missing_frame_json(json_path)
            missing_count += 1

    if cap is not None:
        cap.release()

    logger.info(
        f"Completed 2D pose for {cam_id}: {valid_count} valid detections, "
        f"{missing_count} guarded dummy records saved to {out_dir}"
    )


def process_all_cameras(
    manifest_csv: Union[str, Path],
    session_dir: Union[str, Path],
    output_pose_dir: Union[str, Path] = "pose-2d",
    device: str = "cpu",
) -> None:
    """Processes all cameras found in a session folder against the master manifest."""
    manifest_df = pd.read_csv(manifest_csv)
    session_path = Path(session_dir)

    # Detect camera columns in manifest
    cam_ids = [
        col[:-6] for col in manifest_df.columns if col.endswith("_valid")
    ]
    logger.info(f"Identified {len(cam_ids)} cameras in manifest: {cam_ids}")

    tracker = initialize_rtmpose_tracker(device=device)

    for cam_id in cam_ids:
        # Locate video
        vid_mkv = session_path / f"cameras_{cam_id}.mkv"
        vid_mp4 = session_path / f"cameras_{cam_id}.mp4"
        vid_path = vid_mkv if vid_mkv.is_file() else (vid_mp4 if vid_mp4.is_file() else None)

        if vid_path is None:
            logger.warning(f"No video file found for camera {cam_id} in {session_path}")
            continue

        process_synchronized_stream(
            manifest_csv=manifest_csv,
            cam_id=cam_id,
            video_path=vid_path,
            output_pose_dir=output_pose_dir,
            tracker=tracker,
            device=device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: 2D Multi-View Pose Estimation")
    parser.add_argument("--manifest", type=str, required=True, help="Path to alignment manifest CSV")
    parser.add_argument("--session-dir", type=str, help="HERMES session folder containing cameras_<id>.mkv videos")
    parser.add_argument("--cam-id", type=str, help="Specific camera identifier (e.g. 40478064)")
    parser.add_argument("--video", type=str, help="Path to camera video file")
    parser.add_argument("--frames-dir", type=str, help="Path to extracted frames directory")
    parser.add_argument("--output-pose-dir", type=str, default="pose-2d", help="Destination root for OpenPose JSONs")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.session_dir:
        process_all_cameras(
            manifest_csv=args.manifest,
            session_dir=args.session_dir,
            output_pose_dir=args.output_pose_dir,
            device=args.device,
        )
    elif args.cam_id:
        process_synchronized_stream(
            manifest_csv=args.manifest,
            cam_id=args.cam_id,
            video_path=args.video,
            frames_dir=args.frames_dir,
            output_pose_dir=args.output_pose_dir,
            device=args.device,
        )
    else:
        raise ValueError("Must specify either --session-dir or --cam-id.")


if __name__ == "__main__":
    main()
