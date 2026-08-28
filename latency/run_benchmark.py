#!/usr/bin/env python3

"""
Usage:
    Localhost:
        python run_benchmark.py localhost --device laptop -d 30 -r 2000 -b 100000000 --dry-run
    Multi-device:
        python run_benchmark.py multi-device --device laptop -d 30 -r 2000 -b 100000000 --master-ip 192.168.0.190 --slave-ip 192.168.0.130 --slave-user jetson --slave-dir ~/Documents/hermes-benchmark --slave-os Linux --dry-run
        python run_benchmark.py multi-device --device laptop -d 30 -r 2000 -b 100000000 --master-ip 192.168.0.190 --slave-ip 192.168.0.146 --slave-user Owner --slave-dir D:\\hermes-benchmark --slave-os Windows --dry-run
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def get_hermes_cli_path() -> str:
    python_dir = os.path.dirname(sys.executable)
    for name in ["hermes-cli", "hermes-cli.exe"]:
        candidate = os.path.join(python_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return "hermes-cli"


def run_command(cmd, env=None, dry_run=False):
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"Executing: {cmd_str}")
    if env:
        # print specific environment variables we injected
        injected = {k: v for k, v in env.items() if k.startswith("HERMES_EXP_")}
        if injected:
            print(f"  Env: {injected}")
    if dry_run:
        return

    result = subprocess.run(cmd, env=env, check=True)
    return result


def clean_dir_local(path: Path, dry_run=False):
    print(f"Cleaning local directory: {path}")
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)


def clean_dir_remote(user, host, path, remote_os, dry_run=False):
    print(f"Cleaning remote directory on {host} ({remote_os}): {path}")
    if dry_run:
        return
    if remote_os.lower() == "linux":
        cmd = ["ssh", f"{user}@{host}", f"rm -r '{path}'"]
    else:  # windows
        win_path = path.replace("/", "\\")
        cmd = ["ssh", f"{user}@{host}", f"rmdir /s /q {win_path}"]
    subprocess.run(cmd, check=True)


def handle_localhost(args):
    hermes_cli = get_hermes_cli_path()
    duration = args.duration
    dry_run = args.dry_run

    start_rate = args.start_rate
    start_byte = args.start_byte

    # Grid sweep
    memory_limit = args.mem * 2**30 # 5 GB
    flush_period_s = 10
    rates_grid = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1_000,
        2_000,
        5_000,
    ]
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

    rate_id = rates_grid.index(start_rate)
    byte_id = bytes_grid.index(start_byte)
    start_experiment = (byte_id * len(rates_grid)) + rate_id + 1
    counter = 0

    print("\n=== Starting Localhost Message vs Frequency Sweep of Latency ===")
    total_experiments = len(bytes_grid) * len(rates_grid)
    output_path = Path("data/localhost") / args.device
    for b in bytes_grid:
        for r in rates_grid:
            counter += 1
            if counter < start_experiment:
                continue

            clean_dir_local(output_path / f"bytes_{b}" / f"rate_{r}", dry_run=dry_run)

            print(
                f"\n[{counter}/{total_experiments}]: \tHERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={r}..."
            )

            if (8+8+4) * r * flush_period_s * 2 > memory_limit:
                print(
                    f"[{counter}/{total_experiments}]: HERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={r} exceeds {memory_limit} memory limit."
                )
                continue

            env = os.environ.copy()
            env["HERMES_EXP_FLUSH_PERIOD_S"] = str(flush_period_s)
            env["HERMES_EXP_NUM_BYTES"] = str(b)
            env["HERMES_EXP_RATE"] = str(r)
            env["HERMES_EXP_BUF_LEN"] = str(flush_period_s * r * 2)

            cmd = [
                hermes_cli,
                "-o",
                str(output_path),
                "-d",
                str(duration),
                "--experiment",
                f"bytes={b}",
                f"rate={r}",
                "--config_file",
                "../config/localhost.yml",
            ]
            try:
                run_command(cmd, env=env, dry_run=dry_run)
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment: {e}")
                if not args.continue_on_error:
                    sys.exit(1)


def handle_multi_device(args):
    hermes_cli = get_hermes_cli_path()
    duration = args.duration
    dry_run = args.dry_run

    master_ip = args.master_ip
    slave_ip = args.slave_ip
    slave_user = args.slave_user
    slave_os = args.slave_os
    slave_dir = args.slave_dir

    start_rate = args.start_rate
    start_byte = args.start_byte

    # Grid sweep
    memory_limit = args.mem * 2**30 # 5 GB
    flush_period_s = 10
    rates_grid = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1_000,
        2_000,
        5_000,
    ]
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

    rate_id = rates_grid.index(start_rate)
    byte_id = bytes_grid.index(start_byte)
    start_experiment = (byte_id * len(rates_grid)) + rate_id + 1
    counter = 0

    print("\n=== Starting Multi-Device Message vs Frequency Sweep of Latency ===")
    total_experiments = len(bytes_grid) * len(rates_grid)
    output_path = Path("data/multi_device") / args.device

    for b in bytes_grid:
        for r in rates_grid:
            counter += 1
            # Skip completed tests.
            if counter < start_experiment:
                continue

            clean_dir_local(output_path / f"bytes_{b}" / f"rate_{r}", dry_run=dry_run)

            print(
                f"\n[{counter}/{total_experiments}]: \tHERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={r}..."
            )

            if (8+8+4) * r * flush_period_s * 2 > memory_limit:
                print(
                    f"[{counter}/{total_experiments}]: HERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={r} exceeds {memory_limit} memory limit."
                )
                continue

            env = os.environ.copy()
            env["HERMES_EXP_MASTER_IP"] = str(master_ip)
            env["HERMES_EXP_SLAVE_IP"] = str(slave_ip)
            env["HERMES_EXP_SLAVE_USER"] = str(slave_user)
            env["HERMES_EXP_SLAVE_OS"] = str(slave_os)
            env["HERMES_EXP_SLAVE_PROJ_DIR"] = str(slave_dir)
            env["HERMES_EXP_FLUSH_PERIOD_S"] = str(flush_period_s)
            env["HERMES_EXP_NUM_BYTES"] = str(b)
            env["HERMES_EXP_RATE"] = str(r)
            env["HERMES_EXP_BUF_LEN"] = str(flush_period_s * r * 2)

            # Inject envs
            inject_cmd = [
                sys.executable,
                "../utils/inject_envs.py",
                "../config/slave_src.yml",
                "../config/slave.yml",
            ]
            try:
                run_command(inject_cmd, env=env, dry_run=dry_run)
            except subprocess.CalledProcessError as e:
                print(f"Error injecting environments: {e}")
                if not args.continue_on_error:
                    sys.exit(1)

            # Run master experiment
            cmd = [
                hermes_cli,
                "-o",
                str(output_path),
                "-d",
                str(duration),
                "--experiment",
                f"bytes={b}",
                f"rate={r}",
                "-f",
                "../config/master.yml",
            ]
            try:
                run_command(cmd, env=env, dry_run=dry_run)
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment: {e}")
                if not args.continue_on_error:
                    sys.exit(1)


def handle_plot(args):
    plot_cmd = [sys.executable, "utils/gen_plot_latency.py", args.data_path]
    run_command(plot_cmd)


def main():
    parser = argparse.ArgumentParser(description="HERMES Latency Benchmark runner")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Subcommand to execute"
    )

    # Localhost subcommand
    lh_parser = subparsers.add_parser(
        "localhost", help="Run the localhost latency benchmark sweep"
    )
    lh_parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=60,
        help="Duration for each experiment run in seconds (default: 60)",
    )
    lh_parser.add_argument(
        "-r",
        "--start-rate",
        type=int,
        default=1,
        help="Start rate of the experiment (default: 1)",
    )
    lh_parser.add_argument(
        "-b",
        "--start-byte",
        type=int,
        default=10,
        help="Start bytes of the experiment (default: 10)",
    )
    lh_parser.add_argument(
        "--device", type=str, required=True, help="Name of the device under test"
    )
    lh_parser.add_argument(
        "--mem",
        type=int,
        default=5,
        help="Memory limit for the experiment (in GB)"
    )
    lh_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    lh_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the sweep even if individual runs fail",
    )

    # Multi-device subcommand
    md_parser = subparsers.add_parser(
        "multi-device", help="Run the multi-device latency benchmark sweep"
    )
    md_parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=60,
        help="Duration for each experiment run in seconds (default: 60)",
    )
    md_parser.add_argument(
        "-r",
        "--start-rate",
        type=int,
        default=1,
        help="Start rate of the experiment (default: 1)",
    )
    md_parser.add_argument(
        "-b",
        "--start-byte",
        type=int,
        default=10,
        help="Start bytes of the experiment (default: 10)",
    )
    md_parser.add_argument(
        "--master-ip",
        type=str,
        required=True,
        help="SSH host/IP for the current master device",
    )
    md_parser.add_argument(
        "--slave-ip",
        type=str,
        required=True,
        help="SSH host/IP for the remote slave device",
    )
    md_parser.add_argument(
        "--slave-user",
        type=str,
        required=True,
        help="SSH username for the remote slave device",
    )
    md_parser.add_argument(
        "--slave-dir",
        type=str,
        required=True,
        help="Project directory path on the remote slave device",
    )
    md_parser.add_argument(
        "--slave-os",
        type=str,
        choices=["Windows", "Linux"],
        required=True,
        help="OS type of the remote slave device",
    )
    md_parser.add_argument(
        "--device", type=str, required=True, help="Name of the device under test"
    )
    md_parser.add_argument(
        "--mem",
        type=int,
        default=5,
        help="Memory limit for the experiment (in GB)"
    )
    md_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    md_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the sweep even if individual runs fail",
    )

    # Plot subcommand
    plot_parser = subparsers.add_parser(
        "plot", help="Plot the latency benchmark results"
    )
    plot_parser.add_argument(
        "data_path", type=str, help="Path to the directory containing latency results"
    )

    args = parser.parse_args()

    # Ensure we run from the script's directory so relative paths resolve correctly
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    if args.command == "localhost":
        handle_localhost(args)
    elif args.command == "multi-device":
        handle_multi_device(args)
    elif args.command == "plot":
        handle_plot(args)


if __name__ == "__main__":
    main()
