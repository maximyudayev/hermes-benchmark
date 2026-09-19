#!/usr/bin/env python3
"""
parse_resource_usage.py - Statistical Parser & Profiler for Jetson Resource Logs.

Parses resource logs generated from Jetson devices (e.g. jtop CSV logs) to extract
detailed CPU and GPU metrics, per-core distribution, memory utilization, power,
and workload characteristics.

Usage examples:
    # Print detailed terminal report:
    uv run python parse_resource_usage.py jetson_exo.csv

    # Generate publication-grade diagnostic plot:
    uv run python parse_resource_usage.py jetson_exo.csv --plot jetson_resource_plot.png

    # Export statistical summary to JSON:
    uv run python parse_resource_usage.py jetson_exo.csv --json-out summary.json

    # Export enriched timeseries CSV with per-sample aggregates:
    uv run python parse_resource_usage.py jetson_exo.csv --enriched-csv jetson_exo_enriched.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_cpu_cores(df: pd.DataFrame) -> List[int]:
    """Detect all core IDs present in the dataset (e.g., cpu0, cpu1, ...)."""
    core_ids = set()
    pattern = re.compile(r"^cpu(\d+)_")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            core_ids.add(int(match.group(1)))
    return sorted(list(core_ids))


def normalize_frequency(val: float) -> float:
    """Normalize frequency to MHz (handles values recorded in Hz or kHz)."""
    if pd.isna(val) or val <= 0:
        return 0.0
    # If logged in kHz (e.g. 1,173,000 kHz = 1,173 MHz or 1.173 GHz)
    if val > 50_000:
        return val / 1_000.0
    return float(val)


def compute_series_stats(series: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive statistical distribution for a numerical series."""
    clean = series.dropna()
    if clean.empty:
        return {k: 0.0 for k in ["count", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {
        "count": float(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def parse_jetson_log(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load CSV, compute aggregated columns, and extract comprehensive statistics."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Log file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Timeline & Duration
    if "epoch_s" in df.columns:
        start_epoch = df["epoch_s"].iloc[0]
        df["elapsed_s"] = df["epoch_s"] - start_epoch
    else:
        df["elapsed_s"] = np.arange(len(df))

    duration_s = float(df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0]) if len(df) > 1 else 0.0
    dt_series = df["elapsed_s"].diff().dropna()
    sampling_interval = float(dt_series.median()) if not dt_series.empty else 0.0

    # 2. Normalize Frequencies
    if "gpu_freq_mhz" in df.columns:
        df["gpu_freq_mhz_norm"] = df["gpu_freq_mhz"].apply(normalize_frequency)
    else:
        df["gpu_freq_mhz_norm"] = 0.0

    core_ids = detect_cpu_cores(df)
    for c in core_ids:
        freq_col = f"cpu{c}_freq_mhz"
        if freq_col in df.columns:
            df[f"cpu{c}_freq_mhz_norm"] = df[freq_col].apply(normalize_frequency)

    # 3. Aggregate CPU Statistics Across All Cores
    num_cores = len(core_ids)
    if num_cores > 0:
        # Per-sample average load across all cores
        user_cols = [f"cpu{c}_load_pct_user" for c in core_ids if f"cpu{c}_load_pct_user" in df.columns]
        sys_cols = [f"cpu{c}_load_pct_system" for c in core_ids if f"cpu{c}_load_pct_system" in df.columns]

        df["cpu_user_avg_pct"] = df[user_cols].mean(axis=1) if user_cols else 0.0
        df["cpu_system_avg_pct"] = df[sys_cols].mean(axis=1) if sys_cols else 0.0
        df["cpu_total_avg_pct"] = df["cpu_user_avg_pct"] + df["cpu_system_avg_pct"]

        # Total core-capacity utilized (e.g. 2.1 cores out of 8 cores)
        df["cpu_equivalent_cores"] = (df["cpu_total_avg_pct"] / 100.0) * num_cores

        # Max instantaneous core load
        for c in core_ids:
            u_c = df.get(f"cpu{c}_load_pct_user", 0)
            s_c = df.get(f"cpu{c}_load_pct_system", 0)
            df[f"cpu{c}_load_pct_total"] = u_c + s_c

        tot_cols = [f"cpu{c}_load_pct_total" for c in core_ids]
        df["cpu_peak_core_load_pct"] = df[tot_cols].max(axis=1)

    # 4. Extract Detailed Statistics
    stats: Dict[str, Any] = {
        "metadata": {
            "source_file": str(csv_path.name),
            "sample_count": int(len(df)),
            "duration_seconds": round(duration_s, 2),
            "duration_minutes": round(duration_s / 60.0, 2),
            "sampling_interval_median_s": round(sampling_interval, 3),
            "cpu_core_count": num_cores,
            "start_timestamp": str(df["timestamp"].iloc[0]) if "timestamp" in df.columns else "N/A",
            "end_timestamp": str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else "N/A",
        },
        "gpu": {},
        "cpu_overall": {},
        "cpu_per_core": {},
        "memory": {},
        "power": {},
    }

    # GPU
    if "gpu_load_pct" in df.columns:
        gpu_stats = compute_series_stats(df["gpu_load_pct"])
        gpu_freq_stats = compute_series_stats(df["gpu_freq_mhz_norm"])
        emc_stats = compute_series_stats(df["emc_pct"]) if "emc_pct" in df.columns else {}

        # Activity thresholds
        total_valid = max(1, len(df["gpu_load_pct"].dropna()))
        time_above_10 = float((df["gpu_load_pct"] >= 10.0).sum() / total_valid * 100.0)
        time_above_50 = float((df["gpu_load_pct"] >= 50.0).sum() / total_valid * 100.0)
        time_above_90 = float((df["gpu_load_pct"] >= 90.0).sum() / total_valid * 100.0)

        stats["gpu"] = {
            "utilization_pct": gpu_stats,
            "frequency_mhz": gpu_freq_stats,
            "emc_bus_pct": emc_stats,
            "activity_duty_cycle": {
                "active_gt_10pct_time_pct": round(time_above_10, 2),
                "active_gt_50pct_time_pct": round(time_above_50, 2),
                "saturated_gt_90pct_time_pct": round(time_above_90, 2),
            },
        }

    # Overall CPU
    if num_cores > 0:
        stats["cpu_overall"] = {
            "total_load_pct": compute_series_stats(df["cpu_total_avg_pct"]),
            "user_load_pct": compute_series_stats(df["cpu_user_avg_pct"]),
            "system_load_pct": compute_series_stats(df["cpu_system_avg_pct"]),
            "equivalent_cores_used": compute_series_stats(df["cpu_equivalent_cores"]),
            "peak_single_core_load_pct": compute_series_stats(df["cpu_peak_core_load_pct"]),
        }

        # Per Core Breakdown
        for c in core_ids:
            u_series = df.get(f"cpu{c}_load_pct_user", pd.Series(dtype=float))
            s_series = df.get(f"cpu{c}_load_pct_system", pd.Series(dtype=float))
            tot_series = df.get(f"cpu{c}_load_pct_total", pd.Series(dtype=float))
            freq_series = df.get(f"cpu{c}_freq_mhz_norm", pd.Series(dtype=float))

            stats["cpu_per_core"][f"cpu{c}"] = {
                "total_pct": compute_series_stats(tot_series),
                "user_pct": compute_series_stats(u_series),
                "system_pct": compute_series_stats(s_series),
                "frequency_mhz": compute_series_stats(freq_series),
            }

    # Memory
    if "ram_used_mb" in df.columns:
        stats["memory"]["ram_used_mb"] = compute_series_stats(df["ram_used_mb"])
        if "ram_pct" in df.columns:
            stats["memory"]["ram_pct"] = compute_series_stats(df["ram_pct"])
        if "ram_tot_mb" in df.columns:
            stats["memory"]["ram_total_mb"] = float(df["ram_tot_mb"].iloc[0])

    # Power & Energy
    if "power_cur_mw" in df.columns:
        power_w = df["power_cur_mw"] / 1000.0
        df["power_w"] = power_w
        stats["power"]["power_w"] = compute_series_stats(power_w)

        # Numerical integration for energy: E = sum(P * dt) in Watt-hours
        if len(dt_series) > 0 and len(power_w) > 1:
            dt_clean = df["elapsed_s"].diff().fillna(sampling_interval)
            energy_joules = float((power_w * dt_clean).sum())
            energy_wh = energy_joules / 3600.0
            stats["power"]["total_energy_joules"] = round(energy_joules, 2)
            stats["power"]["total_energy_wh"] = round(energy_wh, 4)

    return df, stats


def print_formatted_report(stats: Dict[str, Any]):
    """Print clean, publication-ready summary tables to terminal."""
    meta = stats["metadata"]
    gpu = stats.get("gpu", {})
    cpu_ov = stats.get("cpu_overall", {})
    cpu_cores = stats.get("cpu_per_core", {})
    mem = stats.get("memory", {})
    pwr = stats.get("power", {})

    print("\n" + "=" * 80)
    print(f" JETSON RESOURCE USAGE PROFILE: {meta['source_file']}")
    print("=" * 80)
    print(f" Duration: {meta['duration_seconds']}s ({meta['duration_minutes']} min) | "
          f"Samples: {meta['sample_count']} (dt ~ {meta['sampling_interval_median_s']}s) | "
          f"CPU Cores: {meta['cpu_core_count']}")
    print(f" Time Window: {meta['start_timestamp']} -> {meta['end_timestamp']}")
    print("-" * 80)

    # 1. EXECUTIVE SUMMARY
    gpu_u = gpu.get("utilization_pct", {})
    cpu_tot = cpu_ov.get("total_load_pct", {})
    cores_used = cpu_ov.get("equivalent_cores_used", {})

    workload_type = "Balanced"
    if gpu_u.get("mean", 0) > 80.0 and cpu_tot.get("mean", 0) < 40.0:
        workload_type = "Heavy GPU-Bound (Inference / Matrix compute saturated)"
    elif cpu_tot.get("mean", 0) > 80.0 and gpu_u.get("mean", 0) < 30.0:
        workload_type = "Heavy CPU-Bound (Host compute / Preprocessing saturated)"
    elif gpu_u.get("mean", 0) > 75.0 and cpu_tot.get("mean", 0) > 60.0:
        workload_type = "Dual Heavy (High GPU + High Multi-Core CPU saturation)"

    print(f"\n[+] WORKLOAD NATURE: {workload_type}")
    print(f"    * GPU Mean Utilization: {gpu_u.get('mean', 0.0):.2f}% (P95: {gpu_u.get('p95', 0.0):.2f}%, Max: {gpu_u.get('max', 0.0):.2f}%)")
    print(f"    * Overall CPU Mean:     {cpu_tot.get('mean', 0.0):.2f}% across {meta['cpu_core_count']} cores (~ {cores_used.get('mean', 0.0):.2f} cores saturated)")
    if mem.get("ram_used_mb"):
        r_u = mem["ram_used_mb"]
        r_tot = mem.get("ram_total_mb", 0.0)
        print(f"    * Memory Allocation:    {r_u.get('mean', 0.0):.0f} MB avg / {r_u.get('max', 0.0):.0f} MB peak (Total: {r_tot:.0f} MB)")
    if pwr.get("power_w"):
        pw = pwr["power_w"]
        print(f"    * Module Power Draw:    {pw.get('mean', 0.0):.2f} W avg (Peak: {pw.get('max', 0.0):.2f} W, Total Energy: {pwr.get('total_energy_wh', 0.0):.4f} Wh)")

    # 2. GPU DETAILED TABLE
    print("\n" + "-" * 80)
    print(" 1. GPU & MEMORY CONTROLLER (EMC) METRICS")
    print("-" * 80)
    print(f"{'Metric':<25} {'Mean':>9} {'Std':>8} {'Min':>8} {'P50':>8} {'P95':>8} {'Max':>8}")
    print("-" * 80)
    if gpu_u:
        print(f"{'GPU Load (%)':<25} {gpu_u['mean']:>9.2f} {gpu_u['std']:>8.2f} {gpu_u['min']:>8.2f} {gpu_u['p50']:>8.2f} {gpu_u['p95']:>8.2f} {gpu_u['max']:>8.2f}")
    if gpu.get("frequency_mhz"):
        gf = gpu["frequency_mhz"]
        print(f"{'GPU Frequency (MHz)':<25} {gf['mean']:>9.1f} {gf['std']:>8.1f} {gf['min']:>8.1f} {gf['p50']:>8.1f} {gf['p95']:>8.1f} {gf['max']:>8.1f}")
    if gpu.get("emc_bus_pct"):
        emc = gpu["emc_bus_pct"]
        print(f"{'EMC Bus Load (%)':<25} {emc['mean']:>9.2f} {emc['std']:>8.2f} {emc['min']:>8.2f} {emc['p50']:>8.2f} {emc['p95']:>8.2f} {emc['max']:>8.2f}")

    if gpu.get("activity_duty_cycle"):
        dc = gpu["activity_duty_cycle"]
        print(f"\nDuty Cycle: >10% Load: {dc['active_gt_10pct_time_pct']}% of time | "
              f">50% Load: {dc['active_gt_50pct_time_pct']}% | "
              f">90% Load (Saturated): {dc['saturated_gt_90pct_time_pct']}%")

    # 3. CPU OVERALL & PER-CORE TABLE
    print("\n" + "-" * 80)
    print(f" 2. CPU UTILIZATION ({meta['cpu_core_count']} CORES)")
    print("-" * 80)
    print(f"{'Component':<12} {'User %':>9} {'Sys %':>9} {'Total %':>9} {'P50 %':>8} {'P95 %':>8} {'Max %':>8} {'Freq (MHz)':>11}")
    print("-" * 80)

    # Aggregated row
    if cpu_ov:
        u_m = cpu_ov["user_load_pct"]["mean"]
        s_m = cpu_ov["system_load_pct"]["mean"]
        tot_m = cpu_ov["total_load_pct"]["mean"]
        tot_p50 = cpu_ov["total_load_pct"]["p50"]
        tot_p95 = cpu_ov["total_load_pct"]["p95"]
        tot_max = cpu_ov["total_load_pct"]["max"]
        print(f"{'TOTAL (Avg)':<12} {u_m:>9.2f} {s_m:>9.2f} {tot_m:>9.2f} {tot_p50:>8.2f} {tot_p95:>8.2f} {tot_max:>8.2f} {'-':>11}")
        print("-" * 80)

    # Per Core rows
    for core_name, core_data in cpu_cores.items():
        u = core_data["user_pct"]["mean"]
        s = core_data["system_pct"]["mean"]
        tot = core_data["total_pct"]
        f = core_data["frequency_mhz"]["mean"]
        print(f"{core_name.upper():<12} {u:>9.2f} {s:>9.2f} {tot['mean']:>9.2f} {tot['p50']:>8.2f} {tot['p95']:>8.2f} {tot['max']:>8.2f} {f:>11.0f}")

    print("=" * 80 + "\n")


def plot_resource_timeseries(df: pd.DataFrame, stats: Dict[str, Any], output_file: Path):
    """Generate high-resolution publication-quality resource timeseries chart."""
    meta = stats["metadata"]
    core_ids = detect_cpu_cores(df)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_gpu, ax_cpu, ax_pwr) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=200)

    t = df["elapsed_s"]

    # 1. GPU Subplot
    if "gpu_load_pct" in df.columns:
        ax_gpu.plot(t, df["gpu_load_pct"], color="#107C41", linewidth=1.5, label="GPU Load (%)")
        ax_gpu.fill_between(t, 0, df["gpu_load_pct"], color="#107C41", alpha=0.15)
    if "emc_pct" in df.columns:
        ax_gpu.plot(t, df["emc_pct"], color="#FF8C00", linestyle="--", linewidth=1.2, label="EMC Bus Load (%)")

    ax_gpu.set_ylabel("Load (%)", fontweight="bold")
    ax_gpu.set_ylim(0, 105)
    ax_gpu.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax_gpu.set_title(f"Jetson Orin Resource Telemetry - {meta['source_file']} ({meta['duration_seconds']}s)",
                     fontsize=12, fontweight="bold", pad=8)

    # 2. CPU Subplot (Individual Cores faint, Aggregate bold)
    color_palette = plt.cm.tab10(np.linspace(0, 1, max(8, len(core_ids))))
    for i, c in enumerate(core_ids):
        col_name = f"cpu{c}_load_pct_total"
        if col_name in df.columns:
            ax_cpu.plot(t, df[col_name], color=color_palette[i % len(color_palette)],
                        linewidth=0.8, alpha=0.4, label=f"Core {c}" if len(core_ids) <= 8 else None)

    if "cpu_total_avg_pct" in df.columns:
        ax_cpu.plot(t, df["cpu_total_avg_pct"], color="#004E8C", linewidth=2.0, label="Overall Avg CPU (%)")

    ax_cpu.set_ylabel("CPU Load (%)", fontweight="bold")
    ax_cpu.set_ylim(0, 105)
    ax_cpu.legend(loc="upper right", ncol=4 if len(core_ids) <= 8 else 2, fontsize=8, frameon=True, framealpha=0.9)

    # 3. Power and Frequencies Subplot
    if "power_w" in df.columns:
        ax_pwr.plot(t, df["power_w"], color="#D83B01", linewidth=1.6, label="Module Power (W)")
        ax_pwr.set_ylabel("Power (Watts)", color="#D83B01", fontweight="bold")
        ax_pwr.tick_params(axis="y", labelcolor="#D83B01")

        # Twin axis for GPU Frequency
        if "gpu_freq_mhz_norm" in df.columns:
            ax_freq = ax_pwr.twinx()
            ax_freq.plot(t, df["gpu_freq_mhz_norm"], color="#6B297A", linestyle=":", linewidth=1.3, label="GPU Clock (MHz)")
            ax_freq.set_ylabel("GPU Clock (MHz)", color="#6B297A", fontweight="bold")
            ax_freq.tick_params(axis="y", labelcolor="#6B297A")
            ax_freq.grid(False)

    ax_pwr.set_xlabel("Elapsed Time (seconds)", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved diagnostic visualization to: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Parser & Profiler for Jetson Resource Logs (GPU, CPU, Memory, Power).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", type=Path, nargs="?", default=Path("jetson_exo.csv"),
                        help="Path to the captured Jetson CSV log file.")
    parser.add_argument("-p", "--plot", type=Path, default=None,
                        help="Generate a diagnostic plot image (.png or .pdf).")
    parser.add_argument("-j", "--json-out", type=Path, default=None,
                        help="Path to export the statistical summary in JSON format.")
    parser.add_argument("-e", "--enriched-csv", type=Path, default=None,
                        help="Path to export the enriched timeseries data with calculated metrics.")

    args = parser.parse_args()

    input_path = args.input_csv
    if not input_path.exists():
        # Fallback to local directory if relative
        alt_path = Path(__file__).resolve().parent / input_path.name
        if alt_path.exists():
            input_path = alt_path
        else:
            print(f"Error: Could not find '{input_path}'", file=sys.stderr)
            sys.exit(1)

    df, stats = parse_jetson_log(input_path)

    # 1. Print formatted terminal report
    print_formatted_report(stats)

    # 2. Export JSON if requested
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[+] Exported statistical summary JSON: {args.json_out.resolve()}")

    # 3. Export Enriched CSV if requested
    if args.enriched_csv:
        df.to_csv(args.enriched_csv, index=False)
        print(f"[+] Exported enriched timeseries CSV: {args.enriched_csv.resolve()}")

    # 4. Generate plot if requested
    if args.plot:
        plot_resource_timeseries(df, stats, args.plot)


if __name__ == "__main__":
    main()
