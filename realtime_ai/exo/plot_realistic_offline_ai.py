"""
Script to process collected data in a live-realistic approach, and plot the segmentation mask.
Currently uses captured data and recorded live inference metadata, to reprocess the same windows.
TODO: add the possibility to reprocess data w.r.t inference latency distribution, instead of just to match live AI. 
Ground truth annotation and smoothing window size (default: 5) are optional.
The `--smooth-window` applies N-size majority vote on raw model outputs to remove prediction chatter. 

Usage:
    python plot_realistic_offline_ai.py \
        --config configs/resnet18_dcl_concat_ln.yml \
        --checkpoint weights/resnet18_dcl_concat_ln_4subj.pt \
        --ai-hdf5 dataset/imu_img/ai_intent.hdf5 \
        --imu-hdf5 dataset/imu_img/ai_intent.hdf5 \
        --ego-hdf5 dataset/imu_img/ai_intent.hdf5 \
        --ego-video dataset/imu_img/smartglasses_ego.mkv \
        --output-dir results/offline/imu_img \
        --device cuda:0 \
        [--annotation dataset/imu_img/annotation.csv] \
        [--smooth-window 30] \
        
"""

import argparse
import os
import time
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch import Tensor, cuda
from torch.nn import Module
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from realtime_ai.exo.utils.config import get_modality_type, load_config
from realtime_ai.exo.utils.transforms import Compose, build_transforms
from realtime_ai.exo.utils.types import Config, ModalityType
from realtime_ai.exo.utils.utils import (
    build_model,
    calculate_kpis,
    fill_annotations_from_csv,
    load_ai_metadata,
    load_imu_data,
    load_video_data,
    load_video_metadata,
    plot_with_smoothed,
    save_inference_results,
    select_imu_columns,
)


def run_inference(
    config: Config,
    modality_type: ModalityType,
    model: Module,
    device: torch.device,
    transforms: dict[ModalityType, Compose | None],
    imu_hdf5_path: Optional[str],
    ego_hdf5_path: Optional[str],
    ego_video_path: Optional[str],
    ai_hdf5_path: Optional[str],
):
    idx_to_label = config.label_mapping["idx_to_label"]

    if ModalityType.RAW_IMU in config.modalities:
        print(f"Loading IMU from {imu_hdf5_path}")
        imu_array_raw, imu_timestamps = load_imu_data(imu_hdf5_path)
        imu_array = select_imu_columns(
            imu_array_raw, config.modalities[ModalityType.RAW_IMU].column_patterns
        )
        imu_window_samples = config.modalities[ModalityType.RAW_IMU].receptive_field
        if transforms[ModalityType.RAW_IMU]:
            imu_array = transforms[ModalityType.RAW_IMU](imu_array)
        print(
            f"  N={len(imu_timestamps)}  span={imu_timestamps[-1] - imu_timestamps[0]:.2f}s"
        )

    if ModalityType.VIDEO in config.modalities:
        print(f"Loading video from {ego_video_path}")
        video_timestamps, video_width, video_height, video_fps, video_num_frames = (
            load_video_metadata(ego_hdf5_path, ego_video_path)
        )
        video_window_samples = config.modalities[ModalityType.VIDEO].receptive_field
        print(
            f"  N={len(video_timestamps)}  span={video_timestamps[-1] - video_timestamps[0]:.2f}s"
        )

    if ModalityType.IMAGE in config.modalities:
        print(f"Loading images from {ego_video_path}")
        image_timestamps, image_width, image_height, image_fps, image_num_frames = (
            load_video_metadata(ego_hdf5_path, ego_video_path)
        )
        print(
            f"  N={len(image_timestamps)}  span={image_timestamps[-1] - image_timestamps[0]:.2f}s"
        )

    ai_window_starts, ai_window_ends, ai_timestamps = load_ai_metadata(ai_hdf5_path)

    inputs = {}
    results = []
    total_ms = 0.0
    t_min = ai_timestamps[0]
    n_inferences = len(ai_timestamps)
    with torch.no_grad():
        for start_s, end_s, toa_s in tqdm(
            zip(ai_window_starts, ai_window_ends, ai_timestamps),
            total=n_inferences,
            desc="Inference instance",
            unit="window",
        ):
            window_start_timestamp = []
            window_end_timestamp = []

            if ModalityType.RAW_IMU in config.modalities:
                window = np.argwhere(
                    (imu_timestamps <= end_s[0]) & (imu_timestamps >= start_s[0])
                )[:, 0]
                chunk = imu_array[window]

                if len(chunk) < imu_window_samples:
                    chunk = np.pad(
                        chunk, ((imu_window_samples - len(chunk), 0), (0, 0))
                    )
                elif len(chunk) > imu_window_samples:
                    chunk = chunk[-imu_window_samples:]

                inputs[ModalityType.RAW_IMU.value] = (
                    torch.from_numpy(np.ascontiguousarray(chunk))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                window_start_timestamp.append(start_s[0])
                window_end_timestamp.append(end_s[0])

            if ModalityType.VIDEO in config.modalities:
                window = np.argwhere(
                    (video_timestamps <= end_s[1]) & (video_timestamps >= start_s[1])
                )

                if len(window) == 0:
                    i0 = i1 = 0
                else:
                    i0 = int(window[0])
                    i1 = int(window[-1])

                chunk = load_video_data(
                    i0, i1, video_fps, video_height, video_width, ego_video_path
                )
                if len(chunk) < video_window_samples:
                    video_array = torch.nn.functional.pad(
                        chunk, (0, 0, 0, 0, 0, 0, video_window_samples - len(chunk), 0)
                    )
                else:
                    video_array = chunk[
                        np.linspace(0, len(chunk) - 1, video_window_samples, dtype=int)
                    ]

                if transforms[ModalityType.VIDEO]:
                    video_array = transforms[ModalityType.VIDEO](video_array)

                inputs[ModalityType.VIDEO.value] = video_array
                window_start_timestamp.append(start_s[1])
                window_end_timestamp.append(end_s[1])

            if ModalityType.IMAGE in config.modalities:
                window = np.argwhere(
                    (image_timestamps <= end_s[1]) & (image_timestamps >= start_s[1])
                )

                if len(window) == 0:
                    i0 = i1 = 0
                else:
                    i0 = window[0].item()
                    i1 = window[-1].item()

                chunk = load_video_data(
                    i0, i1, image_fps, image_height, image_width, ego_video_path
                )
                image_array = chunk[-1:]

                if transforms[ModalityType.IMAGE]:
                    image_array = transforms[ModalityType.IMAGE](image_array)

                inputs[ModalityType.IMAGE.value] = image_array
                window_start_timestamp.append(start_s[1])
                window_end_timestamp.append(end_s[1])

            if device.type == "cuda":
                cuda.synchronize(device)

            t_clock = time.perf_counter()
            if modality_type == ModalityType.MULTIMODAL:
                outputs: Tensor = model(**inputs)
            else:
                outputs: Tensor = model(*inputs.values())

            if device.type == "cuda":
                cuda.synchronize(device)

            elapsed_ms = (time.perf_counter() - t_clock) * 1000
            total_ms += elapsed_ms

            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]

            for h_idx, h in enumerate(config.prediction_horizons):
                pred_idx = int(outputs[h_idx].argmax(dim=-1).item())
                pred_label = idx_to_label.get(str(pred_idx), f"Class_{pred_idx}")
                results.append(
                    {
                        "window_start_timestamp": window_start_timestamp,
                        "window_end_timestamp": window_end_timestamp,
                        "window_end_relative_s": round(toa_s - t_min, 4),
                        "horizon_s": h,
                        "predicted_class": pred_label,
                        "predicted_idx": pred_idx,
                        "true_class": "Unknown",
                        "true_idx": -1,
                        "inference_ms": round(elapsed_ms, 2),
                        "toa_s": toa_s,
                    }
                )

    avg_ms = total_ms / n_inferences
    return results, n_inferences, avg_ms


def main():
    parser = argparse.ArgumentParser(
        description="Validates live AI predictions against offline model ran on the same windows of logged IMU data"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint .pt",
    )
    parser.add_argument(
        "--imu-hdf5",
        required=True,
        help="Path to MVN HDF5 (mvn.hdf5 or ai_intent.hdf5)",
    )
    parser.add_argument(
        "--ego-hdf5",
        help="Path to ego HDF5 (smartglasses.hdf5 or ai_intent.hdf5)",
    )
    parser.add_argument(
        "--ego-video",
        help="Path to ego video file",
    )
    parser.add_argument(
        "--ai-hdf5",
        required=True,
        help="Path to AI HDF5 (ai_intent.hdf5)",
    )
    parser.add_argument(
        "--annotation",
        help="CSV with annotations",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write CSVs and PNGs",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: cuda:0 if available else cpu)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Majority-vote window for smoothed plot",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(
        args.config,
        device=str(device),
        checkpoint_path=os.path.abspath(args.checkpoint),
    )
    modality_type = get_modality_type(config)

    print(
        f"Model: {os.path.basename(args.config)}  |  modality={modality_type}  |  device={device}  |  horizons={config.prediction_horizons}"
    )
    model = build_model(config, device)

    mod_transforms = dict(
        map(
            lambda item: (item[0], build_transforms(item[1].eval_transforms)),
            config.modalities.items(),
        ),
    )

    results, n_windows, avg_ms = run_inference(
        config=config,
        modality_type=modality_type,
        model=model,
        device=device,
        transforms=mod_transforms,
        imu_hdf5_path=args.imu_hdf5,
        ego_hdf5_path=args.ego_hdf5,
        ego_video_path=args.ego_video,
        ai_hdf5_path=args.ai_hdf5,
    )
    if not results:
        print("No predictions produced.")
        return

    df = pd.DataFrame(results)

    if args.annotation:
        df = fill_annotations_from_csv(df, args.annotation)

        kpis_df = calculate_kpis(df)
        kpis_df.insert(0, 'model', 'Offline AI as Live')
        kpis_path = os.path.join(args.output_dir, "kpis.csv")
        kpis_df.to_csv(kpis_path, index=False)
        print(f"KPIs saved to {kpis_path}")

    save_inference_results(
        df,
        args.output_dir,
        prediction_horizons=config.prediction_horizons,
    )

    plot_with_smoothed(
        df, args.output_dir, title="Offline as Live", smooth_window=args.smooth_window
    )

    freq = 1000 / avg_ms if avg_ms > 0 else 0
    print(f"\n{n_windows} windows | avg {avg_ms:.2f} ms | {freq:.2f} Hz")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
