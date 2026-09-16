import argparse
from pathlib import Path
import sys
import numpy as np

# Add parent directory to allow imports from benchmark modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synchronization.utils.cross_correlation import (
    extract_single_window_data,
    run_cross_correlation_analysis,
)
from synchronization.utils.data_preparation import (
    parse_modalities_dict,
    setup_modalities_and_trial,
)
from synchronization.utils.plotting import (
    plot_analysis_and_window,
    print_assessment_summary,
    save_results_to_csv,
)


def parse_time_input(val: str | float) -> float:
    """Parses a time representation (seconds float or MM:SS / HH:MM:SS) into seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    return float(s)


def parse_timestamps_list(raw_items: list[str]) -> list[float]:
    """Flattens and parses comma- or space-separated timestamp strings into floats."""
    tokens = []
    for item in raw_items:
        tokens.extend(item.replace(",", " ").split())
    return [parse_time_input(tok) for tok in tokens]


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate windowed cross-correlation between right hip motor and right thigh Nicla IMU."
    )
    parser.add_argument(
        "--name",
        "-n",
        action="append",
        type=str,
        required=True,
        help="Name of the modality.",
    )
    parser.add_argument(
        "--file",
        "-f",
        action="append",
        type=str,
        required=True,
        help="Path to the HDF5 file.",
    )
    parser.add_argument(
        "--video",
        "-v",
        action="append",
        type=str,
        default=None,
        required=False,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        type=str,
        required=True,
        help="Path to the dataset within the HDF5 file.",
    )
    parser.add_argument(
        "--offset",
        "-o",
        action="append",
        type=float,
        required=True,
        help="Initial offset for alignment.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        required=True,
        help="Path to the output figure (SVG/PNG).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Path to output CSV file. Defaults to saving alongside the SVG with a .csv extension.",
    )
    parser.add_argument(
        "--window-idx",
        type=int,
        default=None,
        help="Index of the window to inspect and plot. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--highlight-timestamps",
        "-t",
        nargs="+",
        default=None,
        metavar="TIMESTAMP",
        help="Two timestamps (in seconds or MM:SS format) to highlight nearest scatter plot points in red.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Window duration in seconds (default: 5.0s).",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=2.5,
        help="Window step in seconds (default: 2.5s for 50%% overlap).",
    )
    parser.add_argument(
        "--resample-rate",
        type=float,
        default=200.0,
        help="Sampling rate to resample signals onto for cross-correlation (default: 200 Hz).",
    )
    parser.add_argument(
        "--min-psd-corr",
        type=float,
        default=0.80,
        help="Minimum PSD correlation threshold to assess signal similarity before cross-correlation (default: 0.80).",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.5,
        help="Minimum cross-correlation peak height to consider valid motion window (default: 0.5).",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=0.5,
        help="Minimum standard deviation (deg) in window to filter static periods (default: 0.5).",
    )
    parser.add_argument(
        "--max-missingness",
        type=float,
        default=0.20,
        help="Maximum allowable normalized missingness rate (0.0 to 1.0) in a window before rejecting it (default: 0.20).",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Plot all windows including stationary/noise periods.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    modalities = parse_modalities_dict(
        names=args.name,
        files=args.file,
        videos=args.video,
        datasets=args.dataset,
        offsets=args.offset,
    )

    (
        motor,
        imu,
        start_trial_toa,
        end_trial_toa,
        motor_timestamps,
        motor_data,
        imu_timestamps,
        imu_data,
    ) = setup_modalities_and_trial(modalities)

    total_windows = int(
        np.floor((end_trial_toa - start_trial_toa - args.window_sec) / args.step_sec)
    ) + 1

    # Parse or prompt for 2 highlight timestamps
    highlight_timestamps = None
    if args.highlight_timestamps is not None:
        parsed_ts = parse_timestamps_list(args.highlight_timestamps)
        if len(parsed_ts) == 2:
            highlight_timestamps = parsed_ts
        else:
            print(
                f"Warning: Expected exactly 2 highlight timestamps, got {len(parsed_ts)}. Skipping timestamp highlight.",
                flush=True,
            )
    else:
        try:
            ts_prompt = input(
                "Enter 2 timestamps to highlight in red (e.g. '120.5 450.0' or '02:00 07:30', or press Enter to skip): "
            ).strip()
            if ts_prompt:
                parsed_ts = parse_timestamps_list([ts_prompt])
                if len(parsed_ts) == 2:
                    highlight_timestamps = parsed_ts
                else:
                    print(
                        f"Warning: Expected exactly 2 highlight timestamps, got {len(parsed_ts)}. Skipping timestamp highlight.",
                        flush=True,
                    )
        except (EOFError, KeyboardInterrupt):
            pass

    # Accept window index from user (via CLI or interactive prompt)
    if args.window_idx is not None:
        selected_window_idx = args.window_idx
    else:
        default_idx = (
            int(np.clip(round(highlight_timestamps[0] / args.step_sec), 0, total_windows - 1))
            if highlight_timestamps is not None
            else 50
        )
        user_input = input(
            f"Enter window index between 0 and {total_windows - 1} (default: {default_idx}): "
        ).strip()
        selected_window_idx = int(user_input) if user_input else default_idx
    assert 0 <= selected_window_idx < total_windows, (
        f"Invalid window index {selected_window_idx}. Must be between 0 and {total_windows - 1}."
    )

    print(f"\nExtracting raw signals and cross-correlation for Window {selected_window_idx}...", flush=True)

    window_data = extract_single_window_data(
        motor_timestamps=motor_timestamps,
        motor_data=motor_data,
        imu_timestamps=imu_timestamps,
        imu_data=imu_data,
        start_trial_toa=start_trial_toa,
        window_idx=selected_window_idx,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        resample_rate=args.resample_rate,
        max_missingness=args.max_missingness,
        min_psd_corr=args.min_psd_corr,
        min_corr=args.min_corr,
        min_std=args.min_std,
    )
    status_msg = "INCLUDED (Passed)" if window_data["is_included"] else f"REJECTED ({window_data['exclusion_reason']})"
    print(
        f"  Window {selected_window_idx} bounds: {window_data['w_start_rel']:.2f}s to {window_data['w_end_rel']:.2f}s\n"
        # f"  Missingness: motor={window_data['motor_missingness']:.1%}, imu={window_data['imu_missingness']:.1%} (max: {window_data['max_missingness']:.1%}, threshold: {args.max_missingness:.1%})\n"
        f"  PSD correlation: {window_data['psd_corr']:.3f} (threshold: {args.min_psd_corr:.2f})\n"
        f"  Peak cross-correlation: {window_data['peak_corr']:.3f} (threshold: {args.min_corr:.2f})\n"
        f"  Estimated relative offset: {window_data['peak_lag_ms']:+.2f} ms (|offset|: {window_data['offset_ms']:.2f} ms)\n"
        f"  Assessment status: {status_msg}"
    )

    print(
        f"\nPerforming windowed assessment & cross-correlation across all {total_windows} windows "
        f"(window={args.window_sec}s, step={args.step_sec}s, resample_rate={args.resample_rate}Hz, max_missingness={args.max_missingness:.1%}, min_psd_corr={args.min_psd_corr})...",
        flush=True,
    )
    analysis_results = run_cross_correlation_analysis(
        motor_timestamps=motor_timestamps,
        motor_data=motor_data,
        imu_timestamps=imu_timestamps,
        imu_data=imu_data,
        start_trial_toa=start_trial_toa,
        end_trial_toa=end_trial_toa,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        resample_rate=args.resample_rate,
        max_missingness=args.max_missingness,
        min_psd_corr=args.min_psd_corr,
        min_corr=args.min_corr,
        min_std=args.min_std,
        include_all=args.include_all,
    )

    print_assessment_summary(analysis_results)

    # Save results to CSV in the same folder as the figure
    csv_out_path = (
        Path(args.out_csv)
        if args.out_csv is not None
        else Path(args.out_file).with_suffix(".csv")
    )
    save_results_to_csv(analysis_results=analysis_results, csv_file=csv_out_path)

    print("\nGenerating composite plot...", flush=True)
    plot_analysis_and_window(
        analysis_results=analysis_results,
        window_data=window_data,
        out_file=args.out_file,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        include_all=args.include_all,
        highlight_timestamps=highlight_timestamps,
    )


if __name__ == "__main__":
    main()
