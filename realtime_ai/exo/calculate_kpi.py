"""
Routine to sweep through recorded prediction CSV files to measure KPIs using calculate_kpis().

Expected CSV columns:
    window_start_timestamp, window_end_timestamp, window_end_relative_s,
    horizon_s, predicted_class, predicted_idx, true_class, true_idx,
    inference_ms, toa_s

Usage examples:
    # Single CSV file
    uv run python realtime_ai/exo/calculate_kpi.py --csv results/live/imu_img/predictions_horizon_0_0s.csv

    # All horizons CSV file
    uv run python realtime_ai/exo/calculate_kpi.py --csv results/live/imu_img/predictions_all_horizons.csv

    # Directory containing prediction CSVs (sweeps all horizon CSVs)
    uv run python realtime_ai/exo/calculate_kpi.py --dir results/live/imu_img --output results/live/imu_img/kpis_summary.csv

    # Sweep smoothing windows to evaluate filtering trade-offs
    uv run python realtime_ai/exo/calculate_kpi.py --csv results/live/imu_img/predictions_all_horizons.csv --sweep-smooth 1 3 5 10 15 20 30

    # Fill ground truth annotations if the CSV was recorded with "Unknown" labels
    uv run python realtime_ai/exo/calculate_kpi.py --csv results/live/imu_img/predictions_all_horizons.csv --annotation dataset/imu_img/annotation.csv
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import Sequence
import pandas as pd

# Add benchmark root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from realtime_ai.exo.utils.utils import (
    calculate_kpis,
    fill_annotations_from_csv,
    plot_f1_matched_horizons,
    smooth_predictions,
)

EXPECTED_COLUMNS = [
    "window_start_timestamp",
    "window_end_timestamp",
    "window_end_relative_s",
    "horizon_s",
    "predicted_class",
    "predicted_idx",
    "true_class",
    "true_idx",
    "inference_ms",
    "toa_s",
]


def load_prediction_csv(
    csv_path: str | Path,
    annotation_path: str | Path | None = None,
    verify_schema: bool = True,
) -> pd.DataFrame:
    """Load a prediction CSV produced by plot_live_ai, plot_offline_ai, or plot_realistic_offline_ai.

    Args:
        csv_path (str | Path): Path to the predictions CSV file.
        annotation_path (str | Path | None): Optional path to ground truth annotations CSV.
        verify_schema (bool): Whether to verify presence of expected columns.

    Returns:
        pd.DataFrame: Loaded DataFrame with prediction data.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if verify_schema:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            print(
                f"Warning: CSV '{csv_path.name}' is missing expected columns: {missing}",
                file=sys.stderr,
            )

    if annotation_path:
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {annotation_path}")
        print(f"Applying annotations from: {annotation_path}")
        df = fill_annotations_from_csv(df, str(annotation_path))

    return df


def measure_kpis(
    df: pd.DataFrame,
    smooth_window: int | None = None,
    include_inference_stats: bool = True,
) -> pd.DataFrame:
    """Compute KPIs for a predictions DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing prediction and ground truth columns.
        smooth_window (int | None): Optional majority vote smoothing window.
        include_inference_stats (bool): If True and 'inference_ms' is present, add mean inference time.

    Returns:
        pd.DataFrame: Computed KPIs per prediction horizon.
    """
    if "true_class" not in df.columns or (df["true_class"] == "Unknown").all():
        print(
            "Warning: Ground truth labels ('true_class') are all 'Unknown'. "
            "KPI metrics cannot be computed without valid ground truth. "
            "Please provide an annotation file via '--annotation'.",
            file=sys.stderr,
        )

    eval_df = df
    if smooth_window is not None and smooth_window > 1:
        eval_df = smooth_predictions(df, window=smooth_window)

    kpis_df = calculate_kpis(eval_df)

    if kpis_df.empty:
        return kpis_df

    if smooth_window is not None and smooth_window > 1:
        kpis_df.insert(1, "smooth_window", smooth_window)

    if include_inference_stats and "inference_ms" in df.columns:
        # Calculate average inference_ms per horizon if available
        avg_inf = (
            df.groupby("horizon_s")["inference_ms"].mean().round(2).to_dict()
            if "horizon_s" in df.columns
            else {"all": round(df["inference_ms"].mean(), 2)}
        )
        kpis_df["avg_inference_ms"] = kpis_df["horizon_s"].map(avg_inf)

    return kpis_df


def sweep_smoothing_windows(
    df: pd.DataFrame,
    smooth_windows: Sequence[int],
) -> pd.DataFrame:
    """Sweep multiple smoothing window sizes over the prediction DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        smooth_windows (Sequence[int]): List of window sizes (e.g. [1, 3, 5, 10, 15]).

    Returns:
        pd.DataFrame: Concatenated KPI results with a 'smooth_window' column.
    """
    all_results = []
    for w in smooth_windows:
        res = measure_kpis(df, smooth_window=w if w > 1 else None)
        if "smooth_window" not in res.columns:
            res.insert(1, "smooth_window", w)
        all_results.append(res)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def sweep_csv_files(
    csv_paths: Sequence[str | Path],
    annotation_path: str | Path | None = None,
    smooth_window: int | None = None,
    smooth_windows: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Sweep through multiple prediction CSV files and measure KPIs for each.

    Args:
        csv_paths (Sequence[str | Path]): List of paths to prediction CSVs.
        annotation_path (str | Path | None): Ground truth annotation file.
        smooth_window (int | None): Single smoothing window.
        smooth_windows (Sequence[int] | None): Sweep over multiple smoothing windows.

    Returns:
        pd.DataFrame: Combined KPI DataFrame with 'source_file' column.
    """
    combined_kpis = []

    for path in csv_paths:
        path = Path(path)
        print(f"\nProcessing: {path}")
        df = load_prediction_csv(path, annotation_path=annotation_path)

        if smooth_windows:
            kpi_res = sweep_smoothing_windows(df, smooth_windows)
        else:
            kpi_res = measure_kpis(df, smooth_window=smooth_window)

        if not kpi_res.empty:
            kpi_res.insert(0, "source_file", path.name)
            combined_kpis.append(kpi_res)
        else:
            print(f"  No valid KPI metrics computed for {path.name} (missing ground truth).")

    if not combined_kpis:
        return pd.DataFrame()

    return pd.concat(combined_kpis, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep through recorded prediction CSV files to measure KPIs using calculate_kpis()."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        help="Path(s) to prediction CSV file(s) (e.g. predictions_horizon_0_0s.csv or predictions_all_horizons.csv)",
    )
    parser.add_argument(
        "--dir",
        help="Directory containing predictions CSV files (will sweep predictions_horizon_*.csv or predictions_all_horizons.csv)",
    )
    parser.add_argument(
        "--annotation",
        help="Optional path to annotations CSV (if ground truth needs to be populated or updated)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Majority-vote smoothing window size (e.g. 5 or 30)",
    )
    parser.add_argument(
        "--sweep-smooth",
        type=int,
        nargs="+",
        help="Sweep across multiple smoothing window sizes (e.g. --sweep-smooth 1 3 5 10 15 20 30)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the computed KPIs as a CSV file",
    )
    parser.add_argument(
        "--plot-f1",
        help="Optional path to save an SVG plot highlighting True Positive segments (100%% saturation) vs False Positives (30%% opacity)",
    )
    parser.add_argument(
        "--f1-k",
        type=float,
        default=0.25,
        help="IoU threshold k for the --plot-f1 visualization (default: 0.25)",
    )
    args = parser.parse_args()

    # Collect target CSV files
    target_files: list[Path] = []

    if args.csv:
        for p in args.csv:
            matches = glob.glob(p)
            if matches:
                target_files.extend(Path(m) for m in matches)
            else:
                target_files.append(Path(p))

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: Directory does not exist: {dir_path}", file=sys.stderr)
            sys.exit(1)

        # Prioritize predictions_all_horizons.csv if present, or all horizon files
        all_horizons_path = dir_path / "predictions_all_horizons.csv"
        if all_horizons_path.exists():
            target_files.append(all_horizons_path)
        else:
            horizon_files = sorted(dir_path.glob("predictions_horizon_*.csv"))
            if horizon_files:
                target_files.extend(horizon_files)
            else:
                all_csvs = sorted(dir_path.glob("*.csv"))
                target_files.extend(all_csvs)

    # Deduplicate while preserving order
    unique_files = list(dict.fromkeys(target_files))

    if not unique_files:
        print(
            "Error: No prediction CSV files specified. Use --csv <path> or --dir <path>.",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    print(f"Found {len(unique_files)} CSV file(s) to process:")
    for f in unique_files:
        print(f"  - {f}")

    # Compute KPIs
    kpi_df = sweep_csv_files(
        csv_paths=unique_files,
        annotation_path=args.annotation,
        smooth_window=args.smooth_window,
        smooth_windows=args.sweep_smooth,
    )

    if kpi_df.empty:
        print("\nNo KPI results could be computed.")
        sys.exit(1)

    print("\n============================== KPI RESULTS ==============================")
    print(kpi_df.to_string(index=False))
    print("=========================================================================\n")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kpi_df.to_csv(output_path, index=False)
        print(f"KPI results saved to: {output_path}")

    if args.plot_f1:
        plot_target = unique_files[0]
        plot_df = load_prediction_csv(plot_target, annotation_path=args.annotation)

        target_path = Path(args.plot_f1)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Derive clean base stem (strip existing _smooth_f1, _smooth, or _f1)
        stem = target_path.stem
        clean_stem = stem
        for suffix in ["_smooth_f1", "_smooth", "_f1"]:
            if clean_stem.endswith(suffix):
                clean_stem = clean_stem[:-len(suffix)]
                break

        # Collect all smoothing windows to plot
        windows_to_plot: list[int] = []
        if args.sweep_smooth:
            windows_to_plot.extend(args.sweep_smooth)
        if args.smooth_window and args.smooth_window not in windows_to_plot:
            windows_to_plot.append(args.smooth_window)

        # 1. Always generate the raw (unsmoothed) F1 plot
        raw_path = target_path.with_name(f"{clean_stem}_f1{target_path.suffix}")
        plot_f1_matched_horizons(
            plot_df,
            save_path=str(raw_path),
            k=args.f1_k,
            title=f"Predictions ({plot_target.name})",
        )

        # 2. Generate an F1 plot for each swept smoothing parameter
        sorted_windows = sorted(set(windows_to_plot))
        canonical_w = args.smooth_window if args.smooth_window else (max(sorted_windows) if sorted_windows and max(sorted_windows) > 1 else None)

        for w in sorted_windows:
            if w <= 1:
                smooth_1_path = target_path.with_name(f"{clean_stem}_smooth_1_f1{target_path.suffix}")
                plot_f1_matched_horizons(
                    plot_df,
                    save_path=str(smooth_1_path),
                    k=args.f1_k,
                    title=f"Predictions ({plot_target.name}, smooth=1)",
                )
                continue

            smoothed_plot_df = smooth_predictions(plot_df, window=w)
            w_path = target_path.with_name(f"{clean_stem}_smooth_{w}_f1{target_path.suffix}")
            plot_f1_matched_horizons(
                smoothed_plot_df,
                save_path=str(w_path),
                k=args.f1_k,
                title=f"Predictions ({plot_target.name}, smooth={w})",
            )

            if canonical_w is not None and w == canonical_w:
                canonical_smooth_path = target_path.with_name(f"{clean_stem}_smooth_f1{target_path.suffix}")
                plot_f1_matched_horizons(
                    smoothed_plot_df,
                    save_path=str(canonical_smooth_path),
                    k=args.f1_k,
                    title=f"Predictions ({plot_target.name}, smooth={w})",
                )


if __name__ == "__main__":
    main()
