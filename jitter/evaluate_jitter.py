import argparse
from io import TextIOWrapper
import h5py
import numpy as np
from pathlib import Path


def unwrap_counter_and_detect_reboots(
    timestamps: np.ndarray,
    raw_indices: np.ndarray,
    nominal_period: float,
):
    times = np.asarray(timestamps, dtype=np.float64)
    indices = np.asarray(raw_indices, dtype=np.int64)
    num_samples = len(indices)

    if num_samples <= 1:
        return (
            times,
            indices,
            np.array([], dtype=np.int64),
            nominal_period if nominal_period else 0.0,
        )

    delta_t = np.diff(times)
    delta_k = np.diff(indices)

    if nominal_period is None or nominal_period <= 0:
        pos_mask = (delta_k > 0) & (delta_t > 0)
        nominal_period = float(np.median(delta_t[pos_mask] / delta_k[pos_mask])) if np.any(pos_mask) else 1.0

    reboot_ids = np.argwhere(delta_k < 0).flatten()

    for gap_id in reboot_ids:
        gap = np.ceil(delta_t[gap_id:gap_id+1] / nominal_period).astype(np.int64)
        indices[gap_id+1:] += indices[gap_id]
        indices[gap_id+1:] += gap

    return times, indices, reboot_ids, nominal_period


def log_jitter(
    log_file: TextIOWrapper,
    file_path: str,
    dataset_path: str,
    sequence_path: str,
    group_name: str = "Default",
    series_name: str = "Sensor",
    nominal_period: float = None,
    mode: str = "nominal",
):
    """
    Reads timestamp data from an HDF5 file, computes inter-sample delay variation
    according to RFC 3393 (difference in timing between consecutive samples),
    prints a summary report, and logs the jitter statistics to CSV.

    Args:
        log_file (TextIOWrapper): Open file handle for writing log information.
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the timestamp dataset within the HDF5 file
                            (e.g., '/my_producer/my_device/toa_s').
        sequence_path (str): Path to the sequence number dataset.
        group_name (str): Host device group name (for CSV logging).
        series_name (str): Sensor / series name (for CSV logging).
        nominal_period (float, optional): Known nominal sampling period in seconds.
        mode (str): 'nominal' for RFC 3393 nominal-referenced delay variation (Method A),
                    or 'consecutive' for cycle-to-cycle second difference (Method B).
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                print(f"Error: Dataset '{dataset_path}' not found in '{file_path}'.")
                print("Available datasets:")

                def print_name(name):
                    print(name)

                f.visit(print_name)
                return None

            timestamps = f[dataset_path][:]
            if sequence_path:
                if sequence_path not in f:
                    print(
                        f"Warning: Sequence dataset '{sequence_path}' not found. Falling back to sequential indices."
                    )
                    indices = None
                else:
                    indices = f[sequence_path][:]
            else:
                indices = None

    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

    if timestamps.ndim > 1:
        timestamps = timestamps.flatten()

    num_samples = len(timestamps)
    if num_samples < 2:
        print("Error: Need at least two data points to compute delay variation.")
        return None

    if indices is None:
        indices = np.arange(num_samples)
    else:
        if indices.ndim > 1:
            indices = indices.flatten()

        if len(indices) != num_samples:
            print(
                f"Warning: Timestamp and counter datasets have different lengths ({num_samples} vs {len(indices)})."
            )
            print("Falling back to sequential indices.")
            indices = np.arange(num_samples)

    nominal_period_arg = nominal_period if (nominal_period is not None and nominal_period > 0) else None

    (
        timestamps,
        corrected_indices,
        reboot_ids,
        nominal_period,
    ) = unwrap_counter_and_detect_reboots(
        timestamps=timestamps,
        raw_indices=indices,
        nominal_period=nominal_period_arg,
    )

    period_s = nominal_period
    sampling_rate_hz = 1.0 / period_s if period_s > 0 else 0.0

    if len(reboot_ids):
        print(
            f"Info [{series_name}]: Detected {len(reboot_ids)} counter reset/reboot transition(s) at index {list(reboot_ids)}. "
            f"Excluded reboot downtime from jitter distribution."
        )

    delta_t = np.diff(timestamps)
    delta_k = np.diff(corrected_indices)

    # Step interval normalized by counter increment
    period_s_dt = delta_t / delta_k

    # Create mask of valid transitions (excluding reboot transitions)
    valid_transitions = np.ones(len(delta_t), dtype=bool)
    if len(reboot_ids) > 0:
        valid_transitions[reboot_ids] = False

    # Calculate RFC 3393 delay variation (jitter)
    if mode == "consecutive":
        # Method B: Cycle-to-cycle timing change (t_{i+1} - t_i)/dk_i - (t_i - t_{i-1})/dk_{i-1}
        # Both adjacent intervals must be valid (neither crossing a reboot)
        valid_consecutive = valid_transitions[:-1] & valid_transitions[1:]
        raw_jitter = np.diff(period_s_dt)
        jitter = raw_jitter[valid_consecutive]
        indices_jitter = corrected_indices[2:][valid_consecutive]
    else:
        # Method A: RFC 3393 Nominal-Referenced IPDV (t_{i+1} - t_i) - delta_k * T_nom
        raw_jitter = delta_t - delta_k * period_s
        jitter = raw_jitter[valid_transitions]
        indices_jitter = corrected_indices[1:][valid_transitions]

    # Convert jitter to milliseconds for reporting and logging
    jitter_ms = jitter * 1000.0

    if len(jitter_ms) == 0:
        print(f"Error: No valid jitter intervals found for {series_name}.")
        return None

    # Calculate statistics (signed distribution with percentiles of absolute magnitude)
    abs_jitter_ms = np.abs(jitter_ms)
    (
        p50_jitter_ms,
        p90_jitter_ms,
        p95_jitter_ms,
        p99_jitter_ms,
    ) = np.percentile(
        abs_jitter_ms, [50, 90, 95, 99]
    )
    mean_jitter_ms = np.mean(jitter_ms)
    std_jitter_ms = np.std(jitter_ms)
    min_jitter_ms = np.min(jitter_ms)
    max_jitter_ms = np.max(jitter_ms)
    peak_to_peak_ms = max_jitter_ms - min_jitter_ms

    # Temporal trend (slope of jitter over time, ms/sample)
    if len(indices_jitter) >= 2:
        jitter_slope, _ = np.polyfit(indices_jitter, jitter_ms, 1)
    else:
        jitter_slope = 0.0

    print("--- Timing Jitter Analysis (RFC 3393) ---")
    print(f"File: {file_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Number of samples: {num_samples} ({len(jitter_ms)} valid transitions)")
    print(f"Estimated sampling rate: {sampling_rate_hz:.4f} Hz")
    print(f"Sampling period (nominal): {period_s * 1000.0:.4f} ms")
    print(f"Mean jitter: {mean_jitter_ms:.4f} ms")
    print(f"Jitter (std dev): {std_jitter_ms:.4f} ms")
    print(f"Median jitter (P50): {p50_jitter_ms:.4f} ms")
    print(f"Jitter (P95 magnitude): {p95_jitter_ms:.4f} ms")
    print(f"Jitter (P99 magnitude): {p99_jitter_ms:.4f} ms")
    print(f"Peak-to-Peak IPDV: {peak_to_peak_ms:.4f} ms")
    print(f"Jitter slope (trend): {jitter_slope:.6e} ms/sample")
    print("-----------------------------------------")

    log_file.write(
        f"{group_name},{series_name},{sampling_rate_hz},{period_s * 1000.0},{len(jitter_ms)},"
        f"{mean_jitter_ms},{std_jitter_ms},{min_jitter_ms},{max_jitter_ms},"
        f"{p50_jitter_ms},{p90_jitter_ms},{p95_jitter_ms},{p99_jitter_ms},"
        f"{peak_to_peak_ms},{jitter_slope}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze timestamp jitter from an HDF5 file using RFC 3393 consecutive sample delay variation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        required=True,
        help="Output folder path. Doesn't have to exist.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to the HDF5 file.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Path to the timestamp dataset.\n"
        "Example: '/MyProducer/MyDevice/toa_s'.",
    )
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        default="",
        help="Path to the sequence dataset (optional).\n"
        "Example: '/MyProducer/MyDevice/counter'.",
    )
    parser.add_argument(
        "--group",
        "-g",
        type=str,
        default="Default",
        help="Host device group name (optional, defaults to 'Default').",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Sensor / series name (optional, defaults to dataset path).",
    )
    parser.add_argument(
        "--period",
        "-p",
        type=float,
        default=None,
        help="Known nominal sampling period in seconds (optional, defaults to median interval).",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=float,
        default=None,
        help="Known nominal sampling rate in Hz (optional, mutually exclusive with --period).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["nominal", "consecutive"],
        default="nominal",
        help="Delay variation formulation:\n"
        "  nominal: Deviation from nominal period (RFC 3393 Method A, default)\n"
        "  consecutive: Consecutive cycle-to-cycle difference (Method B)",
    )

    args = parser.parse_args()

    nominal_period = args.period
    if args.rate is not None and args.rate > 0:
        nominal_period = 1.0 / args.rate

    series_name = args.name if args.name else Path(args.dataset).name
    if not series_name:
        series_name = Path(args.file).stem

    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    with open(Path(out_folder, "stats.csv"), "a") as f:
        if f.tell() == 0:  # Only write header if file is new/empty
            f.write("group,name,rate,period,num_samples,mean,std,min,max,p50,p90,p95,p99,peak_to_peak,jitter_slope\n")
        log_jitter(
            f,
            args.file,
            args.dataset,
            args.sequence,
            group_name=args.group,
            series_name=series_name,
            nominal_period=nominal_period,
            mode=args.mode,
        )
