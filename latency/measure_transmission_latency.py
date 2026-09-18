#!/usr/bin/env python3

import argparse
import time
import random
import multiprocessing as mp
from pathlib import Path
import numpy as np
import zmq
from tqdm import tqdm

from hermes.utils.msgpack_utils import serialize
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


def _broker_process_worker(host: str, ready_queue: mp.Queue, stop_event: mp.Event):
    """Standalone process running the HERMES XPUB/XSUB broker proxy multiplexer."""
    ctx = zmq.Context()

    b_back = ctx.socket(zmq.XSUB)
    b_back.bind(f"tcp://{host}:*")
    port_back = int(b_back.getsockopt_string(zmq.LAST_ENDPOINT).split(":")[-1])

    b_front = ctx.socket(zmq.XPUB)
    b_front.bind(f"tcp://{host}:*")
    port_front = int(b_front.getsockopt_string(zmq.LAST_ENDPOINT).split(":")[-1])

    ready_queue.put((port_back, port_front))

    poller = zmq.Poller()
    poller.register(b_back, zmq.POLLIN)
    poller.register(b_front, zmq.POLLIN)

    while not stop_event.is_set():
        events = dict(poller.poll(50))
        if b_back in events:
            msg = b_back.recv_multipart()
            b_front.send_multipart(msg)
        if b_front in events:
            msg = b_front.recv_multipart()
            b_back.send_multipart(msg)

    b_back.close()
    b_front.close()
    ctx.term()


def _brokered_echo_node_worker(
    host: str,
    port_back: int,
    port_front: int,
    ready_event: mp.Event,
    stop_event: mp.Event,
):
    """Standalone process acting as the recipient pipeline node (echo responder) via broker."""
    ctx = zmq.Context()

    p2_pub = ctx.socket(zmq.PUB)
    p2_pub.connect(f"tcp://{host}:{port_back}")
    p2_sub = ctx.socket(zmq.SUB)
    p2_sub.connect(f"tcp://{host}:{port_front}")
    p2_sub.subscribe(b"probe")

    ready_event.set()

    while not stop_event.is_set():
        if p2_sub.poll(50):
            msg = p2_sub.recv_multipart()
            # Echo payload back to sender on "echo" topic
            p2_pub.send_multipart([b"echo", msg[1]])

    p2_pub.close()
    p2_sub.close()
    ctx.term()


def _direct_echo_node_worker(
    host: str,
    port_fwd: int,
    ready_queue: mp.Queue,
    stop_event: mp.Event,
):
    """Standalone process acting as direct socket echo responder."""
    ctx = zmq.Context()

    p2_pub = ctx.socket(zmq.PUB)
    p2_pub.bind(f"tcp://{host}:*")
    port_rev = int(p2_pub.getsockopt_string(zmq.LAST_ENDPOINT).split(":")[-1])

    p2_sub = ctx.socket(zmq.SUB)
    p2_sub.connect(f"tcp://{host}:{port_fwd}")
    p2_sub.subscribe(b"probe")

    ready_queue.put(port_rev)

    while not stop_event.is_set():
        if p2_sub.poll(50):
            msg = p2_sub.recv_multipart()
            p2_pub.send_multipart([b"echo", msg[1]])

    p2_pub.close()
    p2_sub.close()
    ctx.term()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark pure ZeroMQ transmission latency (one-way = RTT / 2) for HERMES payloads without serdes or DataContainer overhead."
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
        "--mode",
        type=str,
        choices=["brokered", "direct", "both"],
        default="brokered",
        help="Transmission mode: 'brokered' (HERMES XPUB/XSUB broker proxy), 'direct' (raw socket pair), or 'both'",
    )
    parser.add_argument(
        "--payload-type",
        type=str,
        choices=["preserialized", "raw"],
        default="preserialized",
        help="Payload format: 'preserialized' (pre-serialized HERMES probe message) or 'raw' (raw byte buffer)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warm-up packets before timing (default: 20)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP address for binding/connecting sockets (default: '127.0.0.1')",
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
        help="Base directory for output CSVs. Defaults to 'data/transmission/<device>'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of appending",
    )
    return parser.parse_args()


def prepare_payload(num_bytes: int, payload_type: str) -> bytes:
    raw_data = random.randbytes(num_bytes)
    if payload_type == "raw":
        return raw_data
    else:
        # Pre-serialize exact HERMES probe message
        sample = {
            "probe": {
                "data": np.array([[raw_data]], dtype=f"S{num_bytes}"),
                "toa_s": np.array([[get_time()]], dtype=np.float64),
                "sequence": np.array([[0xB00B5]], dtype=np.uint32),
            }
        }
        return serialize(sample)


def run_brokered_benchmark(
    trials: int,
    bytes_grid: list[int],
    payload_type: str,
    warmup: int,
    host: str,
    output_file: Path,
    overwrite: bool = False,
):
    header = "bytes,payload_len,mean,std,min,max,p50,p90,p95,p99\n"
    if overwrite or not output_file.exists():
        with open(output_file, "w") as f:
            f.write(header)

    print(f"\n--- Running Brokered Transmission Benchmark (Standalone Broker Process) ---")
    
    stop_event = mp.Event()
    ready_queue = mp.Queue()
    echo_ready = mp.Event()

    # 1. Start Broker Process
    p_broker = mp.Process(
        target=_broker_process_worker,
        args=(host, ready_queue, stop_event),
        daemon=True,
    )
    p_broker.start()
    port_back, port_front = ready_queue.get()

    # 2. Start Echo Node Process
    p_echo = mp.Process(
        target=_brokered_echo_node_worker,
        args=(host, port_back, port_front, echo_ready, stop_event),
        daemon=True,
    )
    p_echo.start()
    echo_ready.wait()

    # 3. Measurer / Sender (Main Process)
    ctx = zmq.Context()
    p1_pub = ctx.socket(zmq.PUB)
    p1_pub.connect(f"tcp://{host}:{port_back}")
    p1_sub = ctx.socket(zmq.SUB)
    p1_sub.connect(f"tcp://{host}:{port_front}")
    p1_sub.subscribe(b"echo")

    time.sleep(0.4)  # Allow ZMQ slow-joiner subscription synchronization

    try:
        for num_bytes in tqdm(bytes_grid, desc="Brokered bytes"):
            payload = prepare_payload(num_bytes, payload_type)
            actual_len = len(payload)

            # Warmup
            for _ in range(warmup):
                p1_pub.send_multipart([b"probe", payload])
                p1_sub.recv_multipart()

            # Timed iterations
            latencies = np.empty(trials, dtype=np.float64)
            for i in range(trials):
                t0 = time.perf_counter()
                p1_pub.send_multipart([b"probe", payload])
                p1_sub.recv_multipart()
                t1 = time.perf_counter()
                latencies[i] = (t1 - t0) / 2.0  # One-way transmission delay

            p50, p90, p95, p99 = np.percentile(latencies, [50, 90, 95, 99])
            with open(output_file, "a") as f:
                f.write(
                    f"{num_bytes},"
                    f"{actual_len},"
                    f"{np.mean(latencies):.9e},"
                    f"{np.std(latencies):.9e},"
                    f"{np.min(latencies):.9e},"
                    f"{np.max(latencies):.9e},"
                    f"{p50:.9e},{p90:.9e},{p95:.9e},{p99:.9e}\n"
                )

    finally:
        stop_event.set()
        p1_pub.close()
        p1_sub.close()
        ctx.term()

        p_echo.join(timeout=1.0)
        p_broker.join(timeout=1.0)
        if p_echo.is_alive():
            p_echo.terminate()
        if p_broker.is_alive():
            p_broker.terminate()

    print(f"[Done] Brokered transmission results written to: {output_file}")


def run_direct_benchmark(
    trials: int,
    bytes_grid: list[int],
    payload_type: str,
    warmup: int,
    host: str,
    output_file: Path,
    overwrite: bool = False,
):
    header = "bytes,payload_len,mean,std,min,max,p50,p90,p95,p99\n"
    if overwrite or not output_file.exists():
        with open(output_file, "w") as f:
            f.write(header)

    print(f"\n--- Running Direct ZeroMQ Socket Transmission Benchmark (Standalone Echo Process) ---")
    ctx = zmq.Context()

    # Node 1 PUB in main process binds to ephemeral port
    p1_pub = ctx.socket(zmq.PUB)
    p1_pub.bind(f"tcp://{host}:*")
    port_fwd = int(p1_pub.getsockopt_string(zmq.LAST_ENDPOINT).split(":")[-1])

    stop_event = mp.Event()
    ready_queue = mp.Queue()

    p_echo = mp.Process(
        target=_direct_echo_node_worker,
        args=(host, port_fwd, ready_queue, stop_event),
        daemon=True,
    )
    p_echo.start()
    port_rev = ready_queue.get()

    p1_sub = ctx.socket(zmq.SUB)
    p1_sub.connect(f"tcp://{host}:{port_rev}")
    p1_sub.subscribe(b"echo")

    time.sleep(0.4)

    try:
        for num_bytes in tqdm(bytes_grid, desc="Direct bytes"):
            payload = prepare_payload(num_bytes, payload_type)
            actual_len = len(payload)

            for _ in range(warmup):
                p1_pub.send_multipart([b"probe", payload])
                p1_sub.recv_multipart()

            latencies = np.empty(trials, dtype=np.float64)
            for i in range(trials):
                t0 = time.perf_counter()
                p1_pub.send_multipart([b"probe", payload])
                p1_sub.recv_multipart()
                t1 = time.perf_counter()
                latencies[i] = (t1 - t0) / 2.0

            p50, p90, p95, p99 = np.percentile(latencies, [50, 90, 95, 99])
            with open(output_file, "a") as f:
                f.write(
                    f"{num_bytes},"
                    f"{actual_len},"
                    f"{np.mean(latencies):.9e},"
                    f"{np.std(latencies):.9e},"
                    f"{np.min(latencies):.9e},"
                    f"{np.max(latencies):.9e},"
                    f"{p50:.9e},{p90:.9e},{p95:.9e},{p99:.9e}\n"
                )

    finally:
        stop_event.set()
        p1_pub.close()
        p1_sub.close()
        ctx.term()

        p_echo.join(timeout=1.0)
        if p_echo.is_alive():
            p_echo.terminate()

    print(f"[Done] Direct transmission results written to: {output_file}")


def main():
    args = parse_args()
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path("data/transmission") / args.device
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== ZeroMQ Transmission-Only Latency Benchmark ===")
    print(f"Repetitions per size: {args.trials:,}")
    print(f"Payload format: {args.payload_type}")
    print(f"Mode: {args.mode}")
    print(f"Payload sizes (bytes): {args.bytes}")
    print(f"Output directory: {output_path.resolve()}")

    if args.mode in ["brokered", "both"]:
        brokered_csv = output_path / "transmission_brokered_latency.csv"
        run_brokered_benchmark(
            trials=args.trials,
            bytes_grid=args.bytes,
            payload_type=args.payload_type,
            warmup=args.warmup,
            host=args.host,
            output_file=brokered_csv,
            overwrite=args.overwrite,
        )

    if args.mode in ["direct", "both"]:
        direct_csv = output_path / "transmission_direct_latency.csv"
        run_direct_benchmark(
            trials=args.trials,
            bytes_grid=args.bytes,
            payload_type=args.payload_type,
            warmup=args.warmup,
            host=args.host,
            output_file=direct_csv,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
