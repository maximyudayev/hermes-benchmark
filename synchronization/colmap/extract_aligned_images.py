from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_nearest_frame_index(toas: np.ndarray, target_toa_s: float) -> int:
    """Finds the closest frame index for a given `toa_s` timestamp.

    Mirrors the binary search pattern implemented in PysioViz.
    """
    if len(toas) == 0:
        return 0
    idx = int(np.searchsorted(toas, target_toa_s))
    if idx >= len(toas):
        return len(toas) - 1
    if idx > 0 and abs(toas[idx - 1] - target_toa_s) < abs(toas[idx] - target_toa_s):
        idx = idx - 1
    return idx


def read_camera_timestamps(
    h5_path: Union[str, Path],
    dataset_path: Optional[str] = None,
    time_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads frame indices and timestamps in seconds from an HDF5 camera file."""
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        # Strategy 1: Explicit or specified dataset path
        if dataset_path is not None:
            group = h5[dataset_path] if dataset_path in h5 else h5
            return _extract_from_group(group, time_offset)

        # Strategy 2: Check standard flat datasets
        for key in ["toa_s", "timestamps", "timestamp", "time"]:
            if key in h5:
                ts = np.asarray(h5[key]).squeeze()
                if ts.ndim > 1:
                    ts = ts[:, 0]
                frame_idx = (
                    np.asarray(h5["frame_index"]).squeeze()
                    if "frame_index" in h5
                    else np.arange(len(ts))
                )
                if frame_idx.ndim > 1:
                    frame_idx = frame_idx[:, 0]
                return _normalize_timestamps(ts, frame_idx, time_offset)

        # Strategy 3: Auto-discover group with toa_s or frame_timestamp
        candidate_group = _find_timestamp_group(h5)
        if candidate_group is not None:
            return _extract_from_group(candidate_group, time_offset)

        raise KeyError(
            f"Could not find valid timestamp dataset in {h5_path}. "
            f"Available keys: {list(h5.keys())}"
        )


def _find_timestamp_group(h5: Union[h5py.File, h5py.Group]) -> Optional[h5py.Group]:
    """Recursively search for an HDF5 group containing timestamp datasets."""
    for key, item in h5.items():
        if isinstance(item, h5py.Group):
            if "toa_s" in item or "frame_timestamp" in item or "timestamps" in item:
                return item
            sub = _find_timestamp_group(item)
            if sub is not None:
                return sub
    return None


def _extract_from_group(
    group: Union[h5py.File, h5py.Group],
    time_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts timestamp and frame index arrays from an HDF5 group."""
    if "toa_s" in group:
        ts = np.asarray(group["toa_s"]).squeeze()
    elif "frame_timestamp" in group:
        ts = np.asarray(group["frame_timestamp"]).squeeze()
    elif "timestamps" in group:
        ts = np.asarray(group["timestamps"]).squeeze()
    else:
        raise KeyError(f"No timestamp dataset found in group {group.name}")

    if ts.ndim > 1:
        ts = ts[:, 0]

    if "frame_index" in group:
        frame_idx = np.asarray(group["frame_index"]).squeeze()
        if frame_idx.ndim > 1:
            frame_idx = frame_idx[:, 0]
    else:
        frame_idx = np.arange(len(ts))

    return _normalize_timestamps(ts, frame_idx, time_offset)


def _normalize_timestamps(
    ts: np.ndarray,
    frame_idx: np.ndarray,
    time_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts timestamps to seconds and applies optional time offset."""
    ts = ts.astype(np.float64)
    # Detect nanosecond timestamps (UNIX ns > 1e12)
    if np.nanmedian(ts) > 1e12:
        ts = ts / 1e9

    ts = ts - time_offset
    return frame_idx.astype(np.int64), ts


def load_hermes_session(
    session_dir: Union[str, Path],
    camera_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Path]]:
    """Loads camera streams and timestamps from a HERMES raw trial directory.

    Expects:
    - 'cameras.hdf5' containing groups 'cameras/<cam_id>/toa_s'
    - 'cameras_<cam_id>.mkv' (or .mp4) video files

    Returns:
        camera_sources: Mapping of cam_id -> (frame_indices, toas)
        video_paths: Mapping of cam_id -> Path to video file
    """
    session_path = Path(session_dir)
    h5_file = session_path / "cameras.hdf5"
    if not h5_file.is_file():
        raise FileNotFoundError(f"HERMES camera HDF5 file not found: {h5_file}")

    camera_sources: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    video_paths: Dict[str, Path] = {}

    with h5py.File(h5_file, "r") as f:
        available_cams = list(f["cameras"].keys()) if "cameras" in f else []
        target_ids = camera_ids if camera_ids else available_cams

        for cam_id in target_ids:
            toa_path = f"cameras/{cam_id}/toa_s"
            if toa_path not in f:
                logger.warning(f"Dataset {toa_path} not found in {h5_file}. Skipping.")
                continue

            toas = np.asarray(f[toa_path]).ravel().astype(np.float64)
            idx_path = f"cameras/{cam_id}/frame_index"
            if idx_path in f:
                frame_idx = np.asarray(f[idx_path]).ravel().astype(np.int64)
            else:
                frame_idx = np.arange(len(toas), dtype=np.int64)

            # Locate video file (.mkv or .mp4)
            vid_mkv = session_path / f"cameras_{cam_id}.mkv"
            vid_mp4 = session_path / f"cameras_{cam_id}.mp4"
            if vid_mkv.is_file():
                video_paths[cam_id] = vid_mkv
            elif vid_mp4.is_file():
                video_paths[cam_id] = vid_mp4
            else:
                logger.warning(f"No video file found for camera {cam_id} in {session_path}")

            camera_sources[cam_id] = (frame_idx, toas)
            logger.info(
                f"Discovered camera {cam_id}: {len(toas)} frames, "
                f"time range [{toas[0]:.3f}s, {toas[-1]:.3f}s]"
            )

    return camera_sources, video_paths


def build_master_timeline(
    camera_sources: Dict[str, Union[str, Path, Tuple[np.ndarray, np.ndarray]]],
    nominal_fps: float = 30.0,
    min_overlapping_cameras: int = 2,
    time_tolerance_ratio: float = 0.5,
    reference_camera: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    max_duration_s: Optional[float] = None,
) -> pd.DataFrame:
    """Builds a discrete master timeline grid and bins camera frames to slots.

    Args:
        camera_sources: Mapping of camera_id -> HDF5 path or (frame_idx, timestamps) tuple.
        nominal_fps: Target frame rate in Hz.
        min_overlapping_cameras: Minimum number of active cameras required for temporal overlap.
        time_tolerance_ratio: Ratio of dt used as the maximum allowed binning error (default 0.5 * dt).
        reference_camera: Optional camera_id whose first timestamp anchors the grid origin t_0.
        start_time: Optional explicit start timestamp in seconds.
        end_time: Optional explicit end timestamp in seconds.
        max_duration_s: Optional maximum duration in seconds from start_time.

    Returns:
        pd.DataFrame containing master_frame, timestamp, and per-camera indices & valid flags.
    """
    dt = 1.0 / float(nominal_fps)
    max_tolerance = dt * time_tolerance_ratio

    camera_data: Dict[str, pd.DataFrame] = {}

    for cam_id, src in camera_sources.items():
        if isinstance(src, tuple):
            frame_idx, ts = src
        else:
            frame_idx, ts = read_camera_timestamps(src)

        sort_order = np.argsort(ts)
        frame_idx = frame_idx[sort_order]
        ts = ts[sort_order]

        camera_data[cam_id] = pd.DataFrame({"frame_idx": frame_idx, "timestamp": ts})

    if not camera_data:
        raise ValueError("No camera sources provided to build_master_timeline.")

    all_starts = [df["timestamp"].iloc[0] for df in camera_data.values()]
    all_ends = [df["timestamp"].iloc[-1] for df in camera_data.values()]

    global_min_start = (
        max(all_starts)
        if min_overlapping_cameras >= len(camera_data)
        else sorted(all_starts)[min_overlapping_cameras - 1]
    )
    global_max_end = (
        min(all_ends)
        if min_overlapping_cameras >= len(camera_data)
        else sorted(all_ends)[len(all_ends) - min_overlapping_cameras]
    )

    if start_time is None:
        start_time = global_min_start
    elif start_time < 1e8 and global_min_start > 1e8:
        logger.info(
            f"Interpreting start_time {start_time:.3f}s as relative offset from trial start ({global_min_start:.3f}s)"
        )
        start_time = global_min_start + start_time

    if end_time is None:
        end_time = global_max_end
    elif end_time < 1e8 and global_min_start > 1e8:
        logger.info(
            f"Interpreting end_time {end_time:.3f}s as relative offset from trial start ({global_min_start:.3f}s)"
        )
        end_time = global_min_start + end_time

    if max_duration_s is not None and max_duration_s > 0:
        end_time = min(end_time, start_time + max_duration_s)

    if end_time <= start_time:
        raise ValueError(
            f"No temporal overlap found between cameras. "
            f"Starts: {all_starts}, Ends: {all_ends}, Window: [{start_time}, {end_time}]"
        )

    # Anchor grid phase to reference camera if provided
    if reference_camera and reference_camera in camera_data:
        ref_t0 = camera_data[reference_camera]["timestamp"].iloc[0]
        k = int(np.ceil((start_time - ref_t0) / dt))
        start_time = ref_t0 + k * dt

    master_ticks = np.arange(start_time, end_time, dt)
    logger.info(
        f"Master timeline constructed: {len(master_ticks)} frames @ {nominal_fps:.2f} FPS "
        f"({start_time:.3f}s to {end_time:.3f}s, duration: {end_time - start_time:.2f}s)"
    )

    aligned_records = []
    for slot_idx, t_target in enumerate(master_ticks):
        row = {"master_frame": slot_idx, "timestamp": t_target}
        for cam_id, df in camera_data.items():
            ts_arr = df["timestamp"].values
            best_idx = find_nearest_frame_index(ts_arr, t_target)
            err = abs(ts_arr[best_idx] - t_target)

            if err <= max_tolerance:
                row[f"{cam_id}_src_frame"] = int(df["frame_idx"].iloc[best_idx])
                row[f"{cam_id}_valid"] = True
                row[f"{cam_id}_time_error"] = float(err)
            else:
                row[f"{cam_id}_src_frame"] = -1
                row[f"{cam_id}_valid"] = False
                row[f"{cam_id}_time_error"] = float(err)

        aligned_records.append(row)

    manifest_df = pd.DataFrame(aligned_records)
    return manifest_df


def extract_calibration_snapshots(
    video_paths: Dict[str, Union[str, Path]],
    output_dir: Union[str, Path],
    frame_indices_or_toas: Union[List[int], List[float]] = [0],
    toas_per_cam: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, List[Path]]:
    """Extracts reference snapshots from multi-view cameras for interactive extrinsic scene calibration."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_snapshots: Dict[str, List[Path]] = {}

    for cam_id, v_path in video_paths.items():
        v_path = Path(v_path)
        if not v_path.is_file():
            continue

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            continue

        cam_saved = []
        for item in frame_indices_or_toas:
            if isinstance(item, float) and toas_per_cam and cam_id in toas_per_cam:
                f_idx = find_nearest_frame_index(toas_per_cam[cam_id], item)
            else:
                f_idx = int(item)

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                img_path = out_dir / f"{cam_id}_scene_ref_frame_{f_idx:06d}.png"
                cv2.imwrite(str(img_path), frame)
                cam_saved.append(img_path)
                logger.info(f"Saved reference scene snapshot: {img_path}")

        cap.release()
        saved_snapshots[cam_id] = cam_saved

    return saved_snapshots


def synchronize_video_frames(
    manifest_df: pd.DataFrame,
    video_paths: Dict[str, Union[str, Path]],
    output_dir: Union[str, Path],
    image_ext: str = "png",
) -> None:
    """Extracts synchronized frames inserting blank frames on dropped slots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_id, vid_path in video_paths.items():
        vid_path = Path(vid_path)
        if not vid_path.is_file():
            logger.warning(f"Video file not found for {cam_id}: {vid_path}. Skipping extraction.")
            continue

        cam_out_dir = output_dir / cam_id
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video {vid_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

        logger.info(f"Synchronizing frames for {cam_id} ({width}x{height}) into {cam_out_dir}...")

        src_frame_col = f"{cam_id}_src_frame"
        valid_col = f"{cam_id}_valid"

        for _, row in manifest_df.iterrows():
            master_idx = int(row["master_frame"])
            is_valid = bool(row[valid_col])
            src_idx = int(row[src_frame_col])

            out_frame_path = cam_out_dir / f"image{master_idx}.{image_ext}"

            if not is_valid or src_idx < 0:
                cv2.imwrite(str(out_frame_path), blank_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.imwrite(str(out_frame_path), frame)
                else:
                    cv2.imwrite(str(out_frame_path), blank_frame)

        cap.release()
        logger.info(f"Completed extraction for {cam_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Timestamp Harmonization & Grid Alignment")
    parser.add_argument("--session-dir", type=str, help="Path to raw HERMES trial folder containing cameras.hdf5 and .mkv videos")
    parser.add_argument("--cameras", nargs="*", default=[], help="Optional list of cam_id=path_to_h5 pairs")
    parser.add_argument("--videos", nargs="*", default=[], help="Optional list of cam_id=path_to_video pairs")
    parser.add_argument("--fps", type=float, default=30.0, help="Nominal FPS (default: 30.0)")
    parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, default=None, help="End time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Maximum duration in seconds to align")
    parser.add_argument("--reference-cam", type=str, default=None, help="Camera ID to anchor timeline grid phase")
    parser.add_argument("--output-manifest", type=str, default="data/synchronized/alignment_manifest.csv")
    parser.add_argument("--extract-frames", action="store_true", help="Extract video frames with placeholders")
    parser.add_argument("--extract-scene-refs", action="store_true", help="Extract static scene snapshots for extrinsic calibration")
    parser.add_argument("--output-dir", type=str, default="data/synchronized")

    args = parser.parse_args()

    h5_dict = {}
    vid_dict = {}

    if args.session_dir:
        h5_dict, vid_dict = load_hermes_session(args.session_dir)
        if not args.reference_cam and h5_dict:
            args.reference_cam = list(h5_dict.keys())[0]

    for item in args.cameras:
        parts = item.split("=", 1)
        if len(parts) == 2:
            h5_dict[parts[0]] = parts[1]

    for item in args.videos:
        parts = item.split("=", 1)
        if len(parts) == 2:
            vid_dict[parts[0]] = parts[1]

    if not h5_dict:
        raise ValueError("No camera sources specified. Provide --session-dir or --cameras.")

    manifest = build_master_timeline(
        camera_sources=h5_dict,
        nominal_fps=args.fps,
        reference_camera=args.reference_cam,
        start_time=args.start_time,
        end_time=args.end_time,
        max_duration_s=args.duration,
    )

    out_manifest_path = Path(args.output_manifest)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_manifest_path, index=False)
    logger.info(f"Manifest exported to {out_manifest_path}")

    if args.extract_scene_refs and vid_dict:
        toas_map = {cid: src[1] for cid, src in h5_dict.items() if isinstance(src, tuple)}
        extract_calibration_snapshots(
            video_paths=vid_dict,
            output_dir=Path(args.output_dir).parent / "calibration" / "extrinsics",
            frame_indices_or_toas=[0],
            toas_per_cam=toas_map,
        )

    if args.extract_frames and vid_dict:
        synchronize_video_frames(manifest, vid_dict, args.output_dir, "jpg")


if __name__ == "__main__":
    main()
