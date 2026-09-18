#!/usr/bin/env python3

import numpy as np
import h5py
from pathlib import Path
import argparse


def read_hdf5_dataset(
    filename: Path, dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(filename, "r") as f:
        modality = f[dataset_name]
        return np.array(modality["rtt"]), np.array(modality["sequence"]), np.array(modality["toa_s"])


def compute_loss_rate(sequences: np.ndarray) -> float:
    if len(sequences) == 0:
        return 0.0
    seq = sequences.flatten()
    total_expected = int(seq[-1] - seq[0] + 1)
    received = len(seq)
    lost = total_expected - received
    return max(0.0, float(lost) / total_expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=str)

    args = parser.parse_args()

    output_path = Path(args.base_path)

    with open(output_path / "latency.csv", "w") as f:
        if f.tell() == 0:
            f.write("bytes,freq,send_freq,mean,std,min,max,p50,p90,p95,p99\n")

        for bytes_folder in sorted([x for x in output_path.iterdir() if x.is_dir()], key=lambda x: int(x.name.split("_")[1])):
            for rate_folder in sorted([x for x in bytes_folder.iterdir() if x.is_dir()], key=lambda x: int(x.name.split("_")[1])):
                rtts, sequences, toas = read_hdf5_dataset(
                    filename=rate_folder / "probe.hdf5",
                    dataset_name="probe/rtt",
                )

                send_time_s = toas - rtts
                rate_actual = np.polyfit(send_time_s.flatten(), sequences.flatten(), 1)[0]

                lat = (rtts / 2) * 1e3  # convert to ms
                p50, p90, p95, p99 = np.percentile(lat, [50, 90, 95, 99])
                f.write(
                    f"{bytes_folder.name.split('_')[1]},"
                    f"{rate_folder.name.split('_')[1]},"
                    f"{rate_actual:.2f},"
                    f"{np.mean(lat)},"
                    f"{np.std(lat)},"
                    f"{np.min(lat)},"
                    f"{np.max(lat)},"
                    f"{p50},{p90},{p95},{p99}\n"
                )

                # TODO: show time dependent latency distribution, overlay of RTT over time, and loss rate patterns
