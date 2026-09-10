"""
Script to plot segmentation mask of the live AI experiment.
Ground truth annotation and smoothing window size (default: 5) are optional.
The `--smooth-window` applies N-size majority vote on raw model outputs to remove prediction chatter. 

Usage:
    python plot_live_ai.py \
        --hdf5 dataset/imu_img/ai_intent.hdf5 \
        --output-dir results/live/imu_img \
        [--annotation dataset/imu_img/annotation.csv] \
        [--compare-csvs results/offline/imu_img/predictions_all_horizons.csv] \
        [--smooth-window 30] \
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from realtime_ai.exo.utils.utils import (
    calculate_kpis,
    fill_annotations_from_csv,
    load_ai_predictions,
    plot_with_smoothed,
    save_inference_results,
    smooth_predictions
)

def main():
    parser = argparse.ArgumentParser(description="Plots live AIs predictions")
    parser.add_argument(
        "--hdf5",
        required=True,
        help="Path to AI HDF5 (ai_intent.hdf5)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write CSVs and PNGs",
    )
    parser.add_argument(
        "--annotation",
        help="CSV with annotations",
    )
    parser.add_argument(
        "--compare-csvs",
        nargs='+',
        help="Paths to `predictions_all_horizons.csv` of other models for comparison",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Majority-vote window for smoothed plot",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Label mapping
    label_file = Path(__file__).resolve().parent / "configs" / "label_mapping.json"
    with open(label_file, "r") as f:
        label_mapping = json.load(f)
        idx_to_label = label_mapping.get("idx_to_label", {})

    print(f"Loading predictions from {args.hdf5}")
    preds, window_starts, window_ends, timestamps, compute_times = load_ai_predictions(
        args.hdf5
    )
    t_min = float(timestamps[0])
    print(f"  N={len(timestamps)}  span={timestamps[-1] - timestamps[0]:.2f}s")

    results = []
    prediction_horizons = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    for pred_idx, t_start, t_end, timestamp, elapsed_ms in tqdm(
        zip(preds, window_starts, window_ends, timestamps, compute_times / 1000),
        desc="Prediction",
    ):
        for i, h in enumerate(prediction_horizons):
            pred_label = idx_to_label.get(
                str(pred_idx[i].item()), f"Class_{pred_idx[i].item()}"
            )
            results.append(
                {
                    "window_start_timestamp": t_start.tolist(),
                    "window_end_timestamp": t_end.tolist(),
                    "window_end_relative_s": round(timestamp - t_min, 4),
                    "horizon_s": h,
                    "predicted_class": pred_label,
                    "predicted_idx": pred_idx[i].item(),
                    "true_class": "Unknown",
                    "true_idx": -1,
                    "inference_ms": round(elapsed_ms, 2),
                    "toa_s": timestamp,
                }
            )
    df = pd.DataFrame(results)

    if args.annotation:
        df = fill_annotations_from_csv(df, args.annotation)

        df_smooth = smooth_predictions(df, args.smooth_window)
        kpis_df = calculate_kpis(df)

        # kpis_df.insert(0, 'model', 'Live AI')
        # if args.compare_csvs:
        #     for csv_path in args.compare_csvs:
        #         if os.path.exists(csv_path):
        #             other_df = pd.read_csv(csv_path)
        #             if 'true_class' not in other_df.columns or other_df['true_class'].eq('Unknown').all():
        #                 other_df = fill_annotations_from_csv(other_df, args.annotation)
        #             other_kpis = calculate_kpis(other_df)
        #             other_kpis.insert(0, 'model', os.path.basename(os.path.dirname(csv_path)))
        #             kpis_df = pd.concat([kpis_df, other_kpis], ignore_index=True)

        kpis_path = os.path.join(args.output_dir, "kpi_comparison.csv")
        kpis_df.to_csv(kpis_path, index=False)
        print(f"KPIs saved to {kpis_path}")

    save_inference_results(
        df,
        args.output_dir,
        prediction_horizons=prediction_horizons,
    )

    plot_with_smoothed(
        df, args.output_dir, title="Live", smooth_window=args.smooth_window
    )

    compute_times_ms = compute_times * 1000
    avg_ms = compute_times_ms.mean()
    std_ms = compute_times_ms.std()
    freq = 1000 / avg_ms if avg_ms > 0 else 0
    print(
        f"\n{len(timestamps)} windows | avg {avg_ms:.2f} ms | std {std_ms:.2f} ms | {freq:.2f} Hz | {'| '.join(f'{p} {l:.2f} ms' for p, l in zip(['P50', 'P90', 'P95', 'P99'], np.percentile(compute_times_ms, [50, 90, 95, 99])))}"
    )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
