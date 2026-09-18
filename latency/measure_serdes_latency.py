#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

from hermes.utils.msgpack_utils import serialize, deserialize
from hermes.utils.time_utils import get_time


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
        description="Benchmark Msgpack serialization and deserialization latency for HERMES payloads."
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
        help="Base directory for output CSVs. Defaults to 'data/serdes/<device>'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of appending",
    )
    return parser.parse_args()


def benchmark_serdes(trials: int, bytes_grid: list[int], output_path: Path, overwrite: bool = False):
    output_path.mkdir(parents=True, exist_ok=True)
    ser_csv = output_path / "serialization_latency.csv"
    deser_csv = output_path / "deserialization_latency.csv"

    header = "bytes,mean,std,min,max,p50,p90,p95,p99\n"

    # Handle file initialization / overwrite
    if overwrite or not ser_csv.exists():
        with open(ser_csv, "w") as f:
            f.write(header)
    if overwrite or not deser_csv.exists():
        with open(deser_csv, "w") as f:
            f.write(header)

    print(f"\n=== Msgpack Serialization / Deserialization Benchmark ===")
    print(f"Repetitions per size: {trials:,}")
    print(f"Payload sizes (bytes): {bytes_grid}")
    print(f"Output directory: {output_path.resolve()}\n")

    for num_bytes in tqdm(bytes_grid, desc="Payload sizes"):
        # Construct faux payload matching DummyPipeline message format
        toa_s = np.array([[get_time()]], dtype=np.float64)
        raw_data = np.array(
            [[random.randbytes(num_bytes)]],
            dtype=f"S{num_bytes}",
        )

        payload_dict = {
            "probe": {
                "data": raw_data,
                "toa_s": toa_s,
                "sequence": np.array([[0xB00B5]], dtype=np.uint32),
            }
        }

        ser_lat = np.empty(trials, dtype=np.float64)
        deser_lat = np.empty(trials, dtype=np.float64)

        for i in range(trials):
            start_ser = get_time()
            msg = serialize(payload_dict)
            end_ser = get_time()

            deserialize(msg)
            end_deser = get_time()

            ser_lat[i] = end_ser - start_ser
            deser_lat[i] = end_deser - end_ser

        p50_ser, p90_ser, p95_ser, p99_ser = np.percentile(ser_lat, [50, 90, 95, 99])
        with open(ser_csv, "a") as f:
            f.write(
                f"{num_bytes},"
                f"{np.mean(ser_lat):.9e},"
                f"{np.std(ser_lat):.9e},"
                f"{np.min(ser_lat):.9e},"
                f"{np.max(ser_lat):.9e},"
                f"{p50_ser:.9e},{p90_ser:.9e},{p95_ser:.9e},{p99_ser:.9e}\n"
            )

        p50_deser, p90_deser, p95_deser, p99_deser = np.percentile(deser_lat, [50, 90, 95, 99])
        with open(deser_csv, "a") as f:
            f.write(
                f"{num_bytes},"
                f"{np.mean(deser_lat):.9e},"
                f"{np.std(deser_lat):.9e},"
                f"{np.min(deser_lat):.9e},"
                f"{np.max(deser_lat):.9e},"
                f"{p50_deser:.9e},{p90_deser:.9e},{p95_deser:.9e},{p99_deser:.9e}\n"
            )

    print(f"\n[Done] Serialization results written to: {ser_csv}")
    print(f"[Done] Deserialization results written to: {deser_csv}")


def main():
    args = parse_args()
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path("data/serdes") / args.device

    benchmark_serdes(
        trials=args.trials,
        bytes_grid=args.bytes,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
