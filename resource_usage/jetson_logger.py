#!/usr/bin/env python3
"""
jetson_logger.py - High-precision resource logger for Jetson Orin NX using jtop.
Logs timestamp, CPU (loads & freqs per core), GPU, EMC, RAM, and Power to CSV.
"""

import csv
import sys
import time
import argparse
import signal
from datetime import datetime
from jtop import jtop, JtopException

stop_logging = False

def handle_sigint(sig, frame):
    global stop_logging
    print("\nStopping logger...")
    stop_logging = True

def run_logger(output_path: str, interval: float):
    global stop_logging
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print(f"Connecting to jtop daemon (interval: {interval}s)...")
    
    with jtop(interval=interval) as jetson:
        if not jetson.ok():
            print("Error: Could not connect to jtop daemon.", file=sys.stderr)
            sys.exit(1)

        csv_file = open(output_path, mode="w", newline="")
        writer = None

        print(f"Logging metrics to '{output_path}'. Press Ctrl+C or kill process to finish.")

        try:
            while jetson.ok() and not stop_logging:
                # Capture snapshot
                stats = jetson.stats
                cpu_data = jetson.cpu
                gpu_data = jetson.gpu["gpu"]
                ram_data = jetson.memory["RAM"]
                power_data = jetson.power["tot"]

                # Build row
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "epoch_s": round(time.time(), 4),
                    # Memory
                    "ram_used_mb": ram_data["used"] // 1024,
                    "ram_tot_mb": ram_data["tot"] // 1024,
                    "ram_pct": round(ram_data["used"] / ram_data["tot"] * 100, 2),
                    # GPU
                    "gpu_load_pct": gpu_data["status"]["load"],
                    "gpu_freq_mhz": gpu_data["freq"]["cur"],
                    # Memory Controller (EMC)
                    "emc_pct": stats["EMC"],
                    # Total Module Power (mW)
                    "power_cur_mw": power_data["power"],
                    "power_avg_mw": power_data["avg"],
                }

                # Dynamic per-core CPU load and frequency
                for core_id, core_info in enumerate(cpu_data["cpu"]):
                    row[f"cpu{core_id}_load_pct_user"] = round(core_info["user"], 2)
                    row[f"cpu{core_id}_load_pct_system"] = round(core_info["system"], 2)
                    row[f"cpu{core_id}_freq_mhz"] = core_info["freq"]["cur"]

                # Initialize CSV header on first tick
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    writer.writeheader()

                writer.writerow(row)
                csv_file.flush()

                time.sleep(interval)

        except JtopException as e:
            print(f"jtop error: {e}", file=sys.stderr)
        finally:
            csv_file.close()
            print(f"Log successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Jetson Orin NX system resources to CSV.")
    parser.add_argument("-o", "--output", default="jetson_benchmark_metrics.csv", help="CSV output path")
    parser.add_argument("-i", "--interval", type=float, default=0.2, help="Sampling interval in seconds (default: 0.2s)")
    args = parser.parse_args()

    run_logger(output_path=args.output, interval=args.interval)
