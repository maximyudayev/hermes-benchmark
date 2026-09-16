from pathlib import Path
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

matplotlib.rcParams["svg.fonttype"] = "none"


def format_time_tick(s: float, _: int) -> str:
    """Formats seconds into MM:SS format for matplotlib tick formatters."""
    return time.strftime("%M:%S", time.gmtime(max(0, s)))


def format_window_time_tick(s: float, _: int) -> str:
    """Formats window experiment seconds into MM:SS (or MM:SS.s) format for matplotlib tick formatters."""
    sec = max(0.0, s)
    m = int(sec // 60)
    rem_s = sec % 60
    if abs(rem_s - round(rem_s)) < 1e-3:
        return f"{m:02d}:{int(round(rem_s)):02d}"
    else:
        return f"{m:02d}:{rem_s:04.1f}"


def format_time_str(s: float) -> str:
    """Formats seconds into MM:SS.ss string with hundredths precision."""
    sec = max(0.0, s)
    total_hundredths = int(round(sec * 100))
    m = (total_hundredths // 100) // 60
    rem_s = (total_hundredths // 100) % 60
    h_part = total_hundredths % 100
    return f"{m:02d}:{rem_s:02d}.{h_part:02d}"


def save_results_to_csv(analysis_results: dict, csv_file: Path | str) -> None:
    """Logs all computed window results, metrics, and exclusion reasons into a CSV file."""
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_dict = {
        "window_idx": analysis_results["window_indices"],
        "start_time_s": np.round(analysis_results["start_times"], 4),
        "end_time_s": np.round(analysis_results["end_times"], 4),
        "start_toa": analysis_results["start_toas"],
        "end_toa": analysis_results["end_toas"],
        "offset_magnitude_ms": np.round(analysis_results["offsets_ms"], 3),
        "signed_lag_ms": np.round(analysis_results["signed_lags_ms"], 3),
        "peak_correlation": np.round(analysis_results["correlations"], 4),
        "psd_correlation": np.round(analysis_results["psd_correlations"], 4),
    }

    if "motor_missingness" in analysis_results:
        csv_dict["missingness_motor"] = np.round(analysis_results["motor_missingness"], 4)
        csv_dict["missingness_imu"] = np.round(analysis_results["imu_missingness"], 4)
        csv_dict["max_missingness"] = np.round(analysis_results["max_missingness"], 4)

    csv_dict.update(
        {
            "std_motor_deg": np.round(analysis_results["std_motors"], 4),
            "std_imu_deg": np.round(analysis_results["std_imus"], 4),
            "is_included": analysis_results["is_included"],
            "exclusion_reason": analysis_results["exclusion_reasons"],
        }
    )

    df = pd.DataFrame(csv_dict)
    df.to_csv(csv_path, index=False)
    print(f"Computed offset results CSV saved to: {csv_path.resolve()}", flush=True)


def plot_analysis_and_window(
    analysis_results: dict,
    window_data: dict,
    out_file: Path | str,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    include_all: bool = False,
    highlight_timestamps: list[float] | tuple[float, float] | None = None,
) -> None:
    """Plots a 3-panel composite figure:
    1. Raw corresponding window from the two series.
    2. Associated cross-correlation curve for that window.
    3. Full-trial scatter plot of signed offset vs window end timestamp with selected window
       and optional user-specified timestamp points highlighted in red.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (14, 9),
            "lines.markersize": 4,
        }
    )

    fig = plt.figure()
    gs = GridSpec(
        2,
        2,
        height_ratios=[1.05, 1.25],
        hspace=0.35,
        wspace=0.28,
        left=0.07,
        right=0.92,
        top=0.94,
        bottom=0.08,
    )

    # ==========================================
    # Panel 1 (Top Left): Raw Window Time Series
    # ==========================================
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw_twin = ax_raw.twinx()

    m_t = window_data.get("raw_m_t_exp", window_data["raw_m_t"] + window_data["w_start_rel"])
    i_t = window_data.get("raw_i_t_exp", window_data["raw_i_t"] + window_data["w_start_rel"])

    m_line = ax_raw.plot(
        m_t,
        window_data["raw_m_val"],
        "o-",
        color="tab:blue",
        label="Right hip motor",
        markersize=3.5,
        linewidth=1.2,
    )
    i_line = ax_raw_twin.plot(
        i_t,
        window_data["raw_i_val"],
        "s-",
        color="tab:orange",
        label="Right thigh IMU",
        markersize=3.5,
        linewidth=1.2,
    )

    ax_raw.xaxis.set_major_formatter(FuncFormatter(format_window_time_tick))
    ax_raw.set_xlabel("Experiment Time (mm:ss)")
    ax_raw.set_ylabel("Motor position (degrees)", color="tab:blue")
    ax_raw_twin.set_ylabel("IMU angle (degrees)", color="tab:orange")
    ax_raw.tick_params(axis="y", labelcolor="tab:blue")
    ax_raw_twin.tick_params(axis="y", labelcolor="tab:orange")
    ax_raw.set_xlim(window_data["w_start_rel"], window_data["w_end_rel"])
    ax_raw.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Combined legend
    lines = m_line + i_line
    labels = [l.get_label() for l in lines]
    ax_raw.legend(lines, labels, loc="upper right")

    w_idx = window_data["window_idx"]
    w_start_str = format_time_str(window_data["w_start_rel"])
    w_end_str = format_time_str(window_data["w_end_rel"])
    status_str = "INCLUDED" if window_data["is_included"] else f"REJECTED: {window_data['exclusion_reason']}"
    ax_raw.set_title(
        f"Raw Series: Window {w_idx} [{status_str}]\n({w_start_str} - {w_end_str})",
        fontsize=12,
    )

    # ==================================================
    # Panel 2 (Top Right): Window Cross-Correlation Curve
    # ==================================================
    ax_xcorr = fig.add_subplot(gs[0, 1])
    ax_xcorr.plot(
        window_data["lags_ms"],
        window_data["xcorr"],
        color="tab:green",
        linewidth=1.5,
        label="Cross-correlation $R_{xy}(\\tau)$",
    )
    ax_xcorr.axvline(0, color="gray", linestyle=":", linewidth=1.0, label="Zero lag (0 ms)")
    peak_lag = window_data["peak_lag_ms"]
    peak_r = window_data["peak_corr"]
    ax_xcorr.axvline(
        peak_lag,
        color="tab:red",
        linestyle="--",
        linewidth=1.3,
        label=f"Peak lag: {peak_lag:+.1f} ms",
    )
    ax_xcorr.plot(peak_lag, peak_r, "ro", markersize=6)

    ax_xcorr.set_xlabel("Lag $\\tau$ (ms)")
    ax_xcorr.set_ylabel("Normalized Correlation ($R_{xy}$)")
    ax_xcorr.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_xcorr.legend(loc="upper right")
    ax_xcorr.set_title(
        f"Cross-Correlation: Window {w_idx} (Offset: {window_data['peak_lag_ms']:+.1f} ms, $R_{{xy}}$: {peak_r:.3f}, PSD $r$: {window_data['psd_corr']:.3f})",
        fontsize=11,
    )

    # ============================================================
    # Panel 3 (Bottom): Full-Trial Scatter Plot with Window Highlight
    # ============================================================
    ax_scatter = fig.add_subplot(gs[1, :])
    ax_scatter.xaxis.set_major_formatter(FuncFormatter(format_time_tick))

    end_times = analysis_results["end_times"]
    signed_lags_ms = analysis_results["signed_lags_ms"]
    correlations = analysis_results["correlations"]
    is_included = analysis_results["is_included"]
    duration = analysis_results["duration"]

    # Strictly plot only included windows that passed PSD and activity assessment
    included_mask = is_included if len(is_included) > 0 else np.ones(len(end_times), dtype=bool)
    included_lags = signed_lags_ms[included_mask]
    included_end_times = end_times[included_mask]
    included_correlations = correlations[included_mask]

    scatter = ax_scatter.scatter(
        included_end_times,
        included_lags,
        c=included_correlations,
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        s=14,
        alpha=0.65,
        edgecolors="none",
        zorder=3,
        label="Included window signed offset",
    )
    cbar = plt.colorbar(scatter, ax=ax_scatter, pad=0.015)
    cbar.set_label("Peak Cross-Correlation ($R_{xy}$)", rotation=270, labelpad=16)

    # Reference zero offset line
    ax_scatter.axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label="Zero offset (0 ms)",
        zorder=3,
    )

    # Statistics strictly over included windows
    if len(included_lags) > 0:
        median_val = np.median(included_lags)
        ax_scatter.axhline(
            median_val,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"Median signed offset ({median_val:+.1f} ms)",
            zorder=4,
        )

    # Highlight selected window (only plot dot if window was included)
    window_label = (
        f"Selected Window {w_idx} (end: {format_time_str(window_data['w_end_rel'])})"
        if window_data["is_included"]
        else f"Selected Window {w_idx} [REJECTED: {window_data['exclusion_reason']}]"
    )
    ax_scatter.axvline(
        window_data["w_end_rel"],
        color="tab:red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.85,
        label=window_label,
        zorder=5,
    )
    if window_data["is_included"]:
        ax_scatter.plot(
            window_data["w_end_rel"],
            window_data["peak_lag_ms"],
            marker="*",
            color="tab:red",
            markersize=12,
            markeredgecolor="black",
            zorder=6,
        )

    # Highlight points nearest in time to user-provided timestamps
    if highlight_timestamps is not None and len(highlight_timestamps) > 0:
        pts_end_times = included_end_times if len(included_end_times) > 0 else end_times
        pts_lags = included_lags if len(included_lags) > 0 else signed_lags_ms
        pts_corrs = included_correlations if len(included_correlations) > 0 else correlations
        pts_indices = (
            analysis_results["window_indices"][included_mask]
            if len(included_end_times) > 0
            else analysis_results["window_indices"]
        )

        start_toa = (
            analysis_results["start_toas"][0]
            if len(analysis_results.get("start_toas", [])) > 0
            else 0.0
        )

        hl_points = []
        for target_ts in highlight_timestamps:
            target_rel = (
                (target_ts - start_toa)
                if (target_ts >= start_toa and start_toa > 0)
                else target_ts
            )
            diffs = np.abs(pts_end_times - target_rel)
            best_idx = int(np.argmin(diffs))
            hl_points.append(
                {
                    "target": target_rel,
                    "time": float(pts_end_times[best_idx]),
                    "lag": float(pts_lags[best_idx]),
                    "corr": float(pts_corrs[best_idx]),
                    "window_idx": int(pts_indices[best_idx]),
                    "diff": float(diffs[best_idx]),
                }
            )

        if hl_points:
            hl_times = [p["time"] for p in hl_points]
            hl_lags = [p["lag"] for p in hl_points]

            ax_scatter.scatter(
                hl_times,
                hl_lags,
                color="red",
                s=85,
                edgecolors="black",
                linewidths=1.2,
                zorder=7,
                label="Highlighted timestamp points",
            )

            y_min, y_max = ax_scatter.get_ylim()
            y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
            for idx, pt in enumerate(hl_points, 1):
                is_upper = pt["lag"] > (y_min + 0.65 * y_range)
                y_offset = -26 if is_upper else 14
                va = "top" if is_upper else "bottom"

                ax_scatter.annotate(
                    f"P{idx}: {format_time_str(pt['time'])}\n({pt['lag']:+.1f} ms)",
                    xy=(pt["time"], pt["lag"]),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=8.5,
                    fontweight="bold",
                    color="darkred",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="red",
                        linewidth=1.0,
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        lw=1.0,
                    ),
                    zorder=8,
                )

            print("\nHighlighted Scatter Plot Points (nearest to user timestamps):", flush=True)
            for idx, pt in enumerate(hl_points, 1):
                print(
                    f"  Point {idx}: target = {pt['target']:.2f}s ({format_time_str(pt['target'])}) -> "
                    f"nearest point = {pt['time']:.2f}s ({format_time_str(pt['time'])}) [Window {pt['window_idx']}], "
                    f"signed offset = {pt['lag']:+.2f} ms, R_xy = {pt['corr']:.3f} (|dt| = {pt['diff']:.2f}s)",
                    flush=True,
                )

    num_included = int(np.sum(included_mask))
    total_windows = len(included_mask)
    pct_included = 100.0 * num_included / total_windows if total_windows > 0 else 0
    ax_scatter.set_title(
        f"Right Hip Motor vs Right Thigh Nicla IMU Signed Offset Across Trial\n"
        f"({num_included}/{total_windows} windows included, {pct_included:.1f}%)",
        fontsize=13,
    )
    ax_scatter.set_xlabel("Window End Timestamp (mm:ss)")
    ax_scatter.set_ylabel("Signed Relative Offset Δt (ms)")
    ax_scatter.set_xlim(0, duration)
    ax_scatter.set_xticks(np.arange(0, duration + 1, 300))
    ax_scatter.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_scatter.legend(loc="upper right", ncol=2)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Composite plot successfully saved to: {out_path.resolve()}", flush=True)


def compute_r2_scores(y: np.ndarray, baseline: float = 0.0) -> dict[str, float]:
    """Computes R-squared scores for constant (0-slope) models stationed at the mean and median.

    Evaluates:
    1. Uncentered R^2 (relative to zero offset baseline y = 0):
       R^2 = 1 - SS_res / SS_zero
       Quantifies the proportion of variance / deviation from 0 ms explained by the constant shift.
    2. Standard R^2 (relative to sample mean):
       R^2 = 1 - SS_res / SS_tot
    """
    if len(y) == 0:
        return {
            "r2_zero_mean": 0.0,
            "r2_zero_median": 0.0,
            "r2_standard_mean": 0.0,
            "r2_standard_median": 0.0,
        }

    mean_val = float(np.mean(y))
    median_val = float(np.median(y))

    ss_zero = float(np.sum((y - baseline) ** 2))
    ss_tot = float(np.sum((y - mean_val) ** 2))

    ss_res_mean = float(np.sum((y - mean_val) ** 2))
    ss_res_median = float(np.sum((y - median_val) ** 2))

    r2_zero_mean = 1.0 - (ss_res_mean / ss_zero) if ss_zero > 0 else 0.0
    r2_zero_median = 1.0 - (ss_res_median / ss_zero) if ss_zero > 0 else 0.0

    r2_std_mean = 1.0 - (ss_res_mean / ss_tot) if ss_tot > 0 else 0.0
    r2_std_median = 1.0 - (ss_res_median / ss_tot) if ss_tot > 0 else 0.0

    return {
        "r2_zero_mean": r2_zero_mean,
        "r2_zero_median": r2_zero_median,
        "r2_standard_mean": r2_std_mean,
        "r2_standard_median": r2_std_median,
    }


def print_assessment_summary(analysis_results: dict) -> dict[str, float]:
    """Prints terminal summary statistics for window assessment and offset metrics."""
    included_mask = analysis_results["is_included"]
    included_lags = analysis_results["signed_lags_ms"][included_mask]
    included_offsets = analysis_results["offsets_ms"][included_mask]
    reasons = analysis_results["exclusion_reasons"]
    num_passed = int(np.sum(included_mask))
    # num_high_missing = int(np.sum(reasons == "HIGH_MISSINGNESS"))
    num_low_psd = int(np.sum(reasons == "LOW_PSD_CORRELATION"))
    num_stat = int(np.sum(reasons == "STATIONARY"))
    num_low_time = int(np.sum(reasons == "LOW_TIME_CORRELATION"))
    total_w = len(included_mask)

    print(f"\nWindow Assessment Summary ({total_w} total windows):")
    print(f"  Passed (included): {num_passed} ({100.0 * num_passed / total_w:.1f}%)")
    # print(f"  Rejected - High Missingness: {num_high_missing} ({100.0 * num_high_missing / total_w:.1f}%)")
    print(f"  Rejected - Low PSD Correlation: {num_low_psd} ({100.0 * num_low_psd / total_w:.1f}%)")
    print(f"  Rejected - Stationary/Inactive: {num_stat} ({100.0 * num_stat / total_w:.1f}%)")
    print(f"  Rejected - Low Time Correlation: {num_low_time} ({100.0 * num_low_time / total_w:.1f}%)")

    r2_stats = {}
    if len(included_lags) > 0:
        print(f"\nCross-Correlation Signed Offset Statistics ({num_passed} included windows):")
        print(f"  Mean offset: {np.mean(included_lags):+.2f} ms")
        print(f"  SD offset: {np.std(included_lags):+.2f} ms")
        print(f"  Median offset: {np.median(included_lags):+.2f} ms")
        print(f"  90th Percentile offset: {np.percentile(included_lags, 90):+.2f} ms")
        print(f"  95th Percentile offset: {np.percentile(included_lags, 95):+.2f} ms")

        r2_stats = compute_r2_scores(included_lags)
        print(f"\nLinear Model (0 slope) Goodness-of-Fit (R²):")
        print(f"  Stationed at Mean ({np.mean(included_lags):+.2f} ms):")
        print(f"    R² (vs. zero offset baseline): {r2_stats['r2_zero_mean']:.4f}")
        print(f"    R² (standard / vs. mean):     {r2_stats['r2_standard_mean']:.4f}")
        print(f"  Stationed at Median ({np.median(included_lags):+.2f} ms):")
        print(f"    R² (vs. zero offset baseline): {r2_stats['r2_zero_median']:.4f}")
        print(f"    R² (standard / vs. mean):     {r2_stats['r2_standard_median']:.4f}")

        print(f"\nAbsolute Magnitude Statistics:")
        print(f"  Mean absolute magnitude: {np.mean(included_offsets):.2f} ms")
        print(f"  SD absolute magnitude: {np.std(included_offsets):.2f} ms")
        print(f"  Median absolute magnitude: {np.median(included_offsets):.2f} ms")
        print(f"  90th Percentile magnitude: {np.percentile(included_offsets, 90):.2f} ms")
        print(f"  95th Percentile magnitude: {np.percentile(included_offsets, 95):.2f} ms")

    return r2_stats
