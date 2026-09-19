#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['svg.fonttype'] = 'none'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize HERMES pipeline latency breakdown across individual components."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="laptop",
        help="Device name under test (default: 'laptop')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: 'data')",
    )
    parser.add_argument(
        "--transmission-mode",
        type=str,
        choices=["brokered", "direct"],
        default="brokered",
        help="Transmission dataset to load: 'brokered' or 'direct' (default: 'brokered')",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Path to save the generated breakdown figure (e.g. 'breakdown.png')",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive plot window",
    )
    return parser.parse_args()


def load_metric_csv(filepath: Path) -> dict[int, dict[str, float]]:
    """Loads CSV with format: bytes,mean,std,min,max,p50,p90,p95,p99"""
    if not filepath.exists():
        return {}
    results = {}
    with open(filepath, "r") as f:
        header_line = f.readline()
        headers = [h.strip() for h in header_line.split(",")]
        for line in f:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            num_bytes = int(parts[0])
            row = {}
            for col_idx, h in enumerate(headers[1:], start=1):
                if col_idx < len(parts):
                    row[h] = float(parts[col_idx])
            results[num_bytes] = row
    return results


def format_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000} MB" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f} MB"
    elif n >= 1_000:
        return f"{n // 1_000} KB" if n % 1_000 == 0 else f"{n / 1_000:.1f} KB"
    else:
        return f"{n} B"


def analyze_breakdown(device: str, data_dir: Path, tx_mode: str, save_fig: str | None, no_show: bool):
    ser_path = data_dir / "serdes" / device / "serialization_latency.csv"
    deser_path = data_dir / "serdes" / device / "deserialization_latency.csv"
    tx_file = f"transmission_{tx_mode}_latency.csv"
    tx_path = data_dir / "transmission" / device / tx_file
    acq_path = data_dir / "acquisition" / device / "acquisition_latency.csv"
    ingest_path = data_dir / "acquisition" / device / "ingestion_latency.csv"

    # Optional end-to-end latency
    e2e_path = data_dir / "localhost" / device / "latency.csv"

    ser_data = load_metric_csv(ser_path)
    deser_data = load_metric_csv(deser_path)
    tx_data = load_metric_csv(tx_path)
    acq_data = load_metric_csv(acq_path)
    ingest_data = load_metric_csv(ingest_path)
    e2e_data = load_metric_csv(e2e_path)

    # Find common byte sizes available across datasets
    available_datasets = []
    if ser_data:
        available_datasets.append(("Serialization", ser_data))
    if deser_data:
        available_datasets.append(("Deserialization", deser_data))
    if tx_data:
        available_datasets.append((f"Transmission ({tx_mode})", tx_data))
    if acq_data:
        available_datasets.append(("Acquisition", acq_data))
    if ingest_data:
        available_datasets.append(("Ingestion", ingest_data))

    if not available_datasets:
        print(f"\n[Error] No benchmark CSV data found for device '{device}' in {data_dir.resolve()}.")
        print("Please run the benchmark scripts first:")
        print(f"  python latency/measure_serdes_latency.py --device {device}")
        print(f"  python latency/measure_acquisition_latency.py --device {device}")
        print(f"  python latency/measure_transmission_latency.py --device {device} --mode {tx_mode}")
        return

    # Find common byte keys
    common_bytes = sorted(
        set.intersection(*[set(d.keys()) for _, d in available_datasets])
    )

    if not common_bytes:
        # Fallback to union of byte keys
        common_bytes = sorted(
            set.union(*[set(d.keys()) for _, d in available_datasets])
        )

    print(f"\n=========================================================================================")
    print(f"               HERMES Pipeline Latency Component Breakdown ({device.upper()})")
    print(f"=========================================================================================")

    # Table Header
    print(
        f"{'Payload':<10} | {'Acq (ms)':<10} | {'Ser (ms)':<10} | {'Tx (ms)':<10} | {'Deser (ms)':<10} | {'Ingest (ms)':<11} | {'Sum (ms)':<10} | {'Top Bottleneck':<15}"
    )
    print("-" * 97)

    plot_bytes = []
    t_acq = []
    t_ser = []
    t_tx = []
    t_deser = []
    t_ing = []
    t_total = []

    for b in common_bytes:
        v_acq = acq_data.get(b, {}).get("mean", 0.0) * 1e3
        v_ser = ser_data.get(b, {}).get("mean", 0.0) * 1e3
        v_tx = tx_data.get(b, {}).get("mean", 0.0) * 1e3
        v_deser = deser_data.get(b, {}).get("mean", 0.0) * 1e3
        v_ing = ingest_data.get(b, {}).get("mean", 0.0) * 1e3

        total_sum = v_acq + v_ser + v_tx + v_deser + v_ing

        components = {
            "Acquisition": v_acq,
            "Serialization": v_ser,
            "Transmission": v_tx,
            "Deserialization": v_deser,
            "Ingestion": v_ing,
        }
        top_name, top_val = max(components.items(), key=lambda x: x[1])
        pct = (top_val / total_sum * 100) if total_sum > 0 else 0

        print(
            f"{format_bytes(b):<10} | {v_acq:<10.4f} | {v_ser:<10.4f} | {v_tx:<10.4f} | {v_deser:<10.4f} | {v_ing:<11.4f} | {total_sum:<10.4f} | {top_name} ({pct:.1f}%)"
        )

        plot_bytes.append(b)
        t_acq.append(v_acq)
        t_ser.append(v_ser)
        t_tx.append(v_tx)
        t_deser.append(v_deser)
        t_ing.append(v_ing)
        t_total.append(total_sum)

    print("=" * 97)

    # Plotting
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (16, 5),
            "lines.markersize": 5,
        }
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # --- Subplot 1: Component scaling curves (Log-Log) ---
    ax1.plot(plot_bytes, t_acq, marker="o", label="Acquisition", color="#1f77b4")
    ax1.plot(plot_bytes, t_ser, marker="s", label="Serialization (Msgpack)", color="#ff7f0e")
    ax1.plot(plot_bytes, t_tx, marker="^", label=f"Transmission ({tx_mode})", color="#2ca02c")
    ax1.plot(plot_bytes, t_deser, marker="d", label="Deserialization (Msgpack)", color="#d62728")
    ax1.plot(plot_bytes, t_ing, marker="v", label="Ingestion (DataContainer)", color="#9467bd")
    ax1.plot(plot_bytes, t_total, marker="*", linestyle="--", label="Total Decomposed", color="black")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Payload Size (Bytes)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Component Latencies vs. Payload Size")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.legend()

    # --- Subplot 2: Stacked Bar Chart for Selected Sizes (Absolute) ---
    # Select a readable subset across orders of magnitude
    target_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    sub_indices = [i for i, b in enumerate(plot_bytes) if b in target_sizes]
    if not sub_indices:
        # Pick evenly spaced points
        sub_indices = list(range(0, len(plot_bytes), max(1, len(plot_bytes) // 6)))

    sub_labels = [format_bytes(plot_bytes[i]) for i in sub_indices]
    sub_acq = np.array([t_acq[i] for i in sub_indices])
    sub_ser = np.array([t_ser[i] for i in sub_indices])
    sub_tx = np.array([t_tx[i] for i in sub_indices])
    sub_deser = np.array([t_deser[i] for i in sub_indices])
    sub_ing = np.array([t_ing[i] for i in sub_indices])

    x_indices = np.arange(len(sub_labels))
    width = 0.55

    ax2.bar(x_indices, sub_acq, width, label="Acquisition", color="#1f77b4")
    ax2.bar(x_indices, sub_ser, width, bottom=sub_acq, label="Serialization", color="#ff7f0e")
    ax2.bar(x_indices, sub_tx, width, bottom=sub_acq + sub_ser, label=f"Transmission ({tx_mode})", color="#2ca02c")
    ax2.bar(x_indices, sub_deser, width, bottom=sub_acq + sub_ser + sub_tx, label="Deserialization", color="#d62728")
    ax2.bar(x_indices, sub_ing, width, bottom=sub_acq + sub_ser + sub_tx + sub_deser, label="Ingestion", color="#9467bd")

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(sub_labels, rotation=25)
    ax2.set_xlabel("Payload Size")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Pipeline Latency Breakdown (Absolute)")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend(loc="upper left")

    # --- Subplot 3: 100% Stacked Bar Chart (Relative Share) ---
    target_sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    sub_indices = [i for i, b in enumerate(plot_bytes) if b in target_sizes]
    if not sub_indices:
        # Pick evenly spaced points
        sub_indices = list(range(0, len(plot_bytes), max(1, len(plot_bytes) // 6)))

    sub_labels = [format_bytes(plot_bytes[i]) for i in sub_indices]
    sub_acq = np.array([t_acq[i] for i in sub_indices])
    sub_ser = np.array([t_ser[i] for i in sub_indices])
    sub_tx = np.array([t_tx[i] for i in sub_indices])
    sub_deser = np.array([t_deser[i] for i in sub_indices])
    sub_ing = np.array([t_ing[i] for i in sub_indices])

    x_indices = np.arange(len(sub_labels))
    width = 0.55
    sub_total = sub_acq + sub_ser + sub_tx + sub_deser + sub_ing
    safe_total = np.where(sub_total > 0, sub_total, 1.0)

    pct_acq = (sub_acq / safe_total) * 100.0
    pct_ser = (sub_ser / safe_total) * 100.0
    pct_tx = (sub_tx / safe_total) * 100.0
    pct_deser = (sub_deser / safe_total) * 100.0
    pct_ing = (sub_ing / safe_total) * 100.0

    ax3.bar(x_indices, pct_acq, width, label="Acquisition", color="#1f77b4")
    ax3.bar(x_indices, pct_ser, width, bottom=pct_acq, label="Serialization", color="#ff7f0e")
    ax3.bar(x_indices, pct_tx, width, bottom=pct_acq + pct_ser, label=f"Transmission ({tx_mode})", color="#2ca02c")
    ax3.bar(x_indices, pct_deser, width, bottom=pct_acq + pct_ser + pct_tx, label="Deserialization", color="#d62728")
    ax3.bar(x_indices, pct_ing, width, bottom=pct_acq + pct_ser + pct_tx + pct_deser, label="Ingestion", color="#9467bd")

    # Annotate percentage labels inside segments that are large enough (>= 8%)
    layers = [
        (pct_acq, np.zeros_like(pct_acq)),
        (pct_ser, pct_acq),
        (pct_tx, pct_acq + pct_ser),
        (pct_deser, pct_acq + pct_ser + pct_tx),
        (pct_ing, pct_acq + pct_ser + pct_tx + pct_deser),
    ]
    for layer_pct, layer_bottom in layers:
        for idx, (p, b) in enumerate(zip(layer_pct, layer_bottom)):
            if p >= 8.0:
                ax3.text(
                    idx,
                    b + p / 2,
                    f"{p:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )

    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(sub_labels, rotation=25)
    ax3.set_xlabel("Payload Size")
    ax3.set_ylabel("Latency Share (%)")
    ax3.set_ylim(0, 100)
    ax3.set_title("Relative Latency Breakdown (100%)")
    ax3.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))

    # Overlay line on secondary axis showing total latency
    ax3_sec = ax3.twinx()
    ax3_sec.plot(
        x_indices,
        sub_total,
        color="black",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Total Latency",
    )
    ax3_sec.set_yscale("log")
    ax3_sec.set_ylabel("Total Latency (ms)")
    ax3_sec.grid(False)
    ax3_sec.legend(loc="upper left")

    fig.tight_layout()

    if save_fig:
        save_path = Path(save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"\n[Saved] Figure successfully saved to: {save_path.resolve()}")

    if not no_show:
        plt.show()


def main():
    args = parse_args()
    analyze_breakdown(
        device=args.device,
        data_dir=Path(args.data_dir),
        tx_mode=args.transmission_mode,
        save_fig=args.save_fig,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
