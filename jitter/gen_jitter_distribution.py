import argparse
from io import TextIOWrapper
import math
from pathlib import Path
import sys
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams['svg.fonttype'] = 'none'


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


def analyze_jitter(
    log_file: TextIOWrapper,
    group_name: str,
    series_name: str,
    file_path: str,
    dataset_path: str,
    sequence_path: str,
    nominal_rate: float = None,
    mode: str = "nominal",
):
    """
    Reads timestamp data from an HDF5 file, computes inter-sample delay variation
    according to RFC 3393 (difference in timing between consecutive samples),
    and returns the jitter and related statistics for plotting.

    Args:
        log_file (TextIOWrapper): Open file handle for writing log information.
        group_name (str): Name of the group for plotting.
        series_name (str): Name of the series for plotting.
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the timestamp dataset within the HDF5 file
                            (e.g., '/my_producer/my_device/toa_s').
        sequence_path (str): Path to the sequence number dataset.
        nominal_rate (float, optional): Known nominal sampling rate in Hz.
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

    nominal_period_arg = 1.0 / nominal_rate if (nominal_rate is not None and nominal_rate > 0) else None

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

    # Create mask of valid transitions (excluding the reboot transitions)
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
        valid_period_s_dt = period_s_dt[1:][valid_consecutive]
    else:
        # Method A: RFC 3393 Nominal-Referenced IPDV (t_{i+1} - t_i) - delta_k * T_nom
        raw_jitter = delta_t - delta_k * period_s
        jitter = raw_jitter[valid_transitions]
        indices_jitter = corrected_indices[1:][valid_transitions]
        valid_period_s_dt = period_s_dt[valid_transitions]

    # Convert jitter to milliseconds for reporting and plotting
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

    return {
        "residuals_ms": jitter_ms,
        "period_s_dt": valid_period_s_dt,
        "nominal_period_ms": period_s * 1000.0,
        "nominal_rate": sampling_rate_hz,
        "indices": indices_jitter,
        "label": series_name,
        "group": group_name,
    }


def plot_multiple_jitters(out_folder: Path, jitter_data_list: list[dict]):
    """Plots a grid of histograms of timing lag / inter-sample delta for each modality.

    Inspired by Nature Communications (Syntalos, Fig. 4):
    - Plots lag (timing deviation from nominal) on the primary horizontal axis.
    - Displays consecutive sample delta t on a secondary top horizontal axis.
    - Plots proportion of samples on the vertical axis.
    - Includes summary metrics (nominal rate/period, mean +- std, median, P95).
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.markersize": 4,
        }
    )

    num_modalities = len(jitter_data_list)

    # 2 rows, 5 columns grid layout with exact 126:33 aspect ratio
    nrows = 2
    ncols = 5
    scale = 0.16  # 20.16 x 5.28 inches preserves the exact 126 by 33 aspect ratio
    fig_width = 126.0 * scale
    fig_height = 33.0 * scale

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Cohesive color palette for host device groups
    group_palette = {
        "Intel Core i7-9700TE": "#2b5c8f",      # Deep steel blue
        "LattePanda 3 Delta": "#2a9d8f",        # Persian teal
        "AMD Ryzen 7 PRO 8840U": "#7570b3",     # Slate purple
        "Raspberry Pi 5": "#d95f02",            # Burnt orange
    }
    fallback_colors = ["#e7298a", "#66a61e", "#e6ab02", "#1f78b4", "#33a02c", "#a6761d"]
    color_idx = 0
    peak_proportions = []

    # Uniform symmetric horizontal range across all modalities: +/- 30 ms
    x_limit = 30.0
    x_min = -x_limit
    x_max = +x_limit
    # 60 bins spanning 60 ms -> exactly 1.0 ms bin width, with 0.0 ms as center edge
    bins = np.linspace(x_min, x_max, 61)

    for i, data in enumerate(jitter_data_list):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        lags = data["residuals_ms"]
        series_name = data["label"]
        group_name = data["group"]
        t_nom = data.get("nominal_period_ms", 0.0)
        rate_nom = data.get("nominal_rate", 0.0)

        # Assign color based on group
        if group_name in group_palette:
            bar_color = group_palette[group_name]
        else:
            bar_color = fallback_colors[color_idx % len(fallback_colors)]
            group_palette[group_name] = bar_color
            color_idx += 1

        # Count samples outside visible window
        visible_mask = (lags >= x_min) & (lags <= x_max)
        n_outliers = len(lags) - np.sum(visible_mask)

        # Plot histogram with sample proportion weights
        # Use an even number of bins (odd number of edges) so 0.0 is an exact bin edge at center
        weights = np.ones_like(lags) / len(lags)
        hist_counts, _, _ = ax.hist(
            lags,
            bins=bins,
            weights=weights,
            color=bar_color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            zorder=2,
        )
        peak_proportions.append(float(np.max(hist_counts)) if len(hist_counts) > 0 else 0.0)

        # Reference lines: Nominal (0 ms) and Median
        ax.axvline(
            0.0,
            color="#b2182b",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
            label="Nominal (0 ms)",
        )
        med = np.median(lags)
        if abs(med) >= 0.02:
            ax.axvline(
                med,
                color="#333333",
                linestyle=":",
                linewidth=1.1,
                alpha=0.85,
                zorder=3,
                label=f"Median ({med:+.2f} ms)",
            )

        ax.set_xlim(x_min, x_max)
        # ax.set_xlabel("Inter-Sample Lag (ms)", fontsize=8.5)
        # ax.set_ylabel("Proportion of Samples", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelleft=False, left=False)
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        # ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

        # Secondary top x-axis for consecutive sample time delta Δt
        # if t_nom > 0:
        #     sec_ax = ax.secondary_xaxis(
        #         "top",
        #         functions=(lambda x, t=t_nom: x + t, lambda dt, t=t_nom: dt - t),
        #     )
        #     # sec_ax.set_xlabel(r"Time Delta $\Delta t$ (ms)", fontsize=8, fontstyle="italic", labelpad=3)
        #     sec_ax.tick_params(axis="x", labelsize=7.5)

        # Subplot Title
        # ax.set_title(
        #     f"{series_name}\n({group_name})",
        #     fontsize=9.5,
        #     fontweight="bold",
        #     pad=5,
        # )

        # Summary statistics annotation box
        mean_val = np.mean(lags)
        std_val = np.std(lags)
        p50_val = np.median(lags)
        p95_val = np.percentile(np.abs(lags), 95)
        stats_lines = [
            f"$T_{{nom}}$: {t_nom:.2f} ms ({rate_nom:.0f} Hz)",
            f"Mean: {mean_val:+.2f} ± {std_val:.2f} ms",
            f"P50: {p50_val:+.2f} ms | P95: {p95_val:.2f} ms",
            f"$N$: {len(lags):,}",
        ]
        if n_outliers > 0:
            stats_lines.append(f"Outliers: {n_outliers} ({n_outliers / len(lags) * 100:.2f}%)")

        stats_text = "\n".join(stats_lines)
        ax.text(
            0.03,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=7.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.88,
                linewidth=0.7,
            ),
            zorder=4,
        )

        # ax.legend(loc="upper right", frameon=True, framealpha=0.88, fontsize=7, edgecolor="#cccccc")

    # Enforce consistent Y-axis limits across all active subplots
    if peak_proportions:
        max_prop_global = max(peak_proportions)
        global_ymax = min(1.05, max(0.1, math.ceil((max_prop_global * 1.18) * 20) / 20.0))
        for i in range(num_modalities):
            r = i // ncols
            c = i % ncols
            axes[r, c].set_ylim(0.0, global_ymax)

    # Hide any unused subplots in the grid
    for j in range(num_modalities, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_visible(False)

    # fig.suptitle(
    #     "Inter-Sample Delay Variation Distributions Across Modalities (RFC 3393)",
    #     fontsize=12,
    #     fontweight="bold",
    #     y=0.985,
    # )
    fig.tight_layout()

    out_svg = out_folder / "jitter_distribution.svg"
    out_png = out_folder / "jitter_distribution.png"
    plt.savefig(out_svg)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved jitter distribution plots to:\n  {out_svg}\n  {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot inter-sample delay variation (RFC 3393) from an HDF5 file.",
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
        "--group",
        "-g",
        action="append",
        type=str,
        required=True,
        help="Grouping by host device name. Can be specified multiple times.",
    )
    parser.add_argument(
        "--name",
        "-n",
        action="append",
        type=str,
        required=True,
        help="Name for the dataset. Can be specified multiple times.",
    )
    parser.add_argument(
        "--file",
        "-f",
        action="append",
        type=str,
        required=True,
        help="Path to the HDF5 file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        type=str,
        required=True,
        help="Path to the timestamp dataset. Must be specified for each file.\n"
        "Example: '/MyProducer/MyDevice/toa_s'. Can be specified multiple times.",
    )
    parser.add_argument(
        "--sequence",
        "-s",
        action="append",
        type=str,
        required=True,
        help="Path to the sequence dataset. Can be specified multiple times (pass \"\" if none).\n"
        "Example: '/MyProducer/MyDevice/counter'.",
    )
    parser.add_argument(
        "--nominal-rate",
        "-r",
        action="append",
        type=float,
        required=True,
        help="Nominal sampling rate of the modality.",
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

    if (
        len(args.sequence) != len(args.file)
        or len(args.dataset) != len(args.file)
        or len(args.name) != len(args.file)
        or len(args.group) != len(args.file)
        or len(args.nominal_rate) != len(args.file)
    ):
        sys.exit(
            "Error: The number of --group, --name, --file, --dataset, and --sequence arguments must match."
        )

    jitter_data_list = []
    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    with open(Path(out_folder, "stats.csv"), "w") as f:
        f.write("group,name,rate,period,num_samples,mean,std,min,max,p50,p90,p95,p99,peak_to_peak,jitter_slope\n")
        for i in range(len(args.file)):
            data = analyze_jitter(
                log_file=f,
                group_name=args.group[i],
                series_name=args.name[i],
                file_path=args.file[i],
                dataset_path=args.dataset[i],
                sequence_path=args.sequence[i],
                nominal_rate=args.nominal_rate[i],
                mode=args.mode,
            )
            if data:
                jitter_data_list.append(data)

    if jitter_data_list:
        plot_multiple_jitters(out_folder, jitter_data_list)
    else:
        print("No valid data could be analyzed. Exiting.")
