#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

from hermes.utils.time_utils import get_time
from hermes.dummy.data_container import DummyPipeDataContainer


DEFAULT_BYTES_GRID = [
    10,
    20,
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Data Acquisition and Storage (DataContainer shared memory push) latency in HERMES."
    )
    parser.add_argument(
        "-n",
        "--trials",
        type=int,
        default=1_000,
        help="Number of repetitions per payload size (default: 1000)",
    )
    parser.add_argument(
        "-b",
        "--bytes",
        type=int,
        nargs="+",
        default=DEFAULT_BYTES_GRID,
        help="List of payload sizes in bytes to evaluate (e.g. -b 100 1000 10000 100000 1000000)",
    )
    parser.add_argument(
        "--buf-len",
        type=int,
        default=10,
        help="Circular buffer length for DataContainer allocation (default: 2000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="laptop",
        help="Device under test identifier for output organization (default: 'laptop')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for output CSVs. Defaults to 'data/acquisition/<device>'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of appending",
    )
    return parser.parse_args()


def benchmark_acquisition(
    trials: int,
    bytes_grid: list[int],
    buf_len: int,
    output_path: Path,
    overwrite: bool = False,
):
    output_path.mkdir(parents=True, exist_ok=True)
    acq_csv = output_path / "acquisition_latency.csv"
    shm_push_csv = output_path / "shm_push_latency.csv"
    capture_csv = output_path / "capture_latency.csv"
    ingest_csv = output_path / "ingestion_latency.csv"

    header = "bytes,mean,std,min,max,p50,p90,p95,p99\n"

    for file_path in [acq_csv, shm_push_csv, capture_csv, ingest_csv]:
        if overwrite or not file_path.exists():
            with open(file_path, "w") as f:
                f.write(header)

    print(f"\n=== Data Acquisition & Shared Memory Push Benchmark ===")
    print(f"Repetitions per size: {trials:,}")
    print(f"Circular buffer length: {buf_len:,}")
    print(f"Payload sizes (bytes): {bytes_grid}")
    print(f"Output directory: {output_path.resolve()}\n")

    for num_bytes in tqdm(bytes_grid, desc="Payload sizes"):
        # Pre-allocate random raw data template
        raw_template = np.array(
            [[random.randbytes(num_bytes)]],
            dtype=f"S{num_bytes}",
        )

        container_out = None
        container_in = None

        try:
            container_out = DummyPipeDataContainer(
                sampling_rate_hz=1000,
                payload_num_bytes=num_bytes,
                buf_len=buf_len,
            )
            container_in = DummyPipeDataContainer(
                sampling_rate_hz=1000,
                payload_num_bytes=num_bytes,
                buf_len=buf_len,
            )

            capture_lats = np.empty(trials, dtype=np.float64)
            push_out_lats = np.empty(trials, dtype=np.float64)
            push_in_lats = np.empty(trials, dtype=np.float64)
            total_acq_lats = np.empty(trials, dtype=np.float64)

            seq_arr = np.array([[0]], dtype=np.uint32)

            for i in range(trials):
                # 1. Measure Capture / Sample Generation Delay
                # TODO: factor out into a separate benchmark?
                t0 = get_time()
                toa_s = np.array([[t0]], dtype=np.float64)
                # Slicing/copying the array buffer to simulate sample acquisition
                sample_data = raw_template.copy()
                seq_arr[0, 0] = i
                payload_dict = {
                    "probe": {
                        "data": sample_data,
                        "toa_s": toa_s,
                        "sequence": seq_arr.copy(),
                    }
                }
                t1 = get_time()
                capture_lats[i] = t1 - t0

                # 2. Measure Outgoing DataContainer Shared Memory Push
                container_out.push(process_time_s=t1, data=payload_dict)
                t2 = get_time()
                push_out_lats[i] = t2 - t1

                # Total acquisition delay for the producer
                total_acq_lats[i] = t2 - t0

                # 3. Measure Consumer Ingestion Push (recipient side)
                container_in.push(process_time_s=t2, data=payload_dict)
                t3 = get_time()
                push_in_lats[i] = t3 - t2

            def write_stats(filepath: Path, data: np.ndarray):
                p50, p90, p95, p99 = np.percentile(data, [50, 90, 95, 99])
                with open(filepath, "a") as f:
                    f.write(
                        f"{num_bytes},"
                        f"{np.mean(data):.9e},"
                        f"{np.std(data):.9e},"
                        f"{np.min(data):.9e},"
                        f"{np.max(data):.9e},"
                        f"{p50:.9e},{p90:.9e},{p95:.9e},{p99:.9e}\n"
                    )

            write_stats(acq_csv, total_acq_lats)
            write_stats(shm_push_csv, push_out_lats)
            write_stats(capture_csv, capture_lats)
            write_stats(ingest_csv, push_in_lats)

        finally:
            if container_out is not None:
                container_out.close_all()
                container_out.unlink_all()
            if container_in is not None:
                container_in.close_all()
                container_in.unlink_all()

    print(f"\n[Done] Total Acquisition results: {acq_csv}")
    print(f"[Done] Shared Memory Push results: {shm_push_csv}")
    print(f"[Done] Capture / Prep results:     {capture_csv}")
    print(f"[Done] Ingestion Push results:     {ingest_csv}")


def main():
    args = parse_args()
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path("data/acquisition") / args.device

    benchmark_acquisition(
        trials=args.trials,
        bytes_grid=args.bytes,
        buf_len=args.buf_len,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
