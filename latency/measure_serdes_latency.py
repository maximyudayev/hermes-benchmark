#!/usr/bin/env python3

from pathlib import Path
from hermes.utils.msgpack_utils import deserialize
from hermes.utils.msgpack_utils import serialize
from hermes.utils.msgpack_utils import *
from hermes.utils.time_utils import get_time
import numpy as np 
import random
from tqdm import tqdm


if __name__ == "__main__":
    num_trials = 1_000
    bytes_grid = [
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

    for num_bytes in tqdm(bytes_grid):
        toa_s = np.array([[get_time()]], dtype=np.float64)
        data = np.array(
            [[random.randbytes(num_bytes)]],
            dtype=f"S{num_bytes}",
        )

        data = {
            "probe": {
                "data": data,
                "toa_s": toa_s,
                "sequence": np.array([[0xB00B5]], dtype=np.uint32),
            }
        }

        ser_lat = []
        deser_lat = []
        for i in range(num_trials):
            start_ser_s = get_time()
            msg = serialize(data)
            end_ser_s = get_time()
            deserialize(msg)
            end_deser_s = get_time()
            ser_lat.append(end_ser_s - start_ser_s)
            deser_lat.append(end_deser_s - end_ser_s)

        output_path = Path("data/serdes")
        with open(output_path / "serialization_latency.csv", "a") as f:
            if f.tell() == 0:
                f.write("bytes,mean,std,min,max,p50,p90,p95,p99\n")

            p50, p90, p95, p99 = np.percentile(ser_lat, [50, 90, 95, 99])
            f.write(
                f"{num_bytes},"
                f"{np.mean(ser_lat)},"
                f"{np.std(ser_lat)},"
                f"{np.min(ser_lat)},"
                f"{np.max(ser_lat)},"
                f"{p50},{p90},{p95},{p99}\n"
            )

        with open(output_path / "deserialization_latency.csv", "a") as f:
            if f.tell() == 0:
                f.write("bytes,mean,std,min,max,p50,p90,p95,p99\n")

            p50, p90, p95, p99 = np.percentile(deser_lat, [50, 90, 95, 99])
            f.write(
                f"{num_bytes},"
                f"{np.mean(deser_lat)},"
                f"{np.std(deser_lat)},"
                f"{np.min(deser_lat)},"
                f"{np.max(deser_lat)},"
                f"{p50},{p90},{p95},{p99}\n"
            )
