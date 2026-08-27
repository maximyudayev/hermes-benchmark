#!/usr/bin/env python3
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

    # Grid sweep
    memory_limit = 5*2**30 # 5 GB
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
        1000,
        2000,
        5000,
        10_000,
        20_000,
        50_000,
        100_000,
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

    print("\n=== Starting Localhost Message vs Frequency Sweep of Latency ===")
    counter = 1
    total_experiments = len(bytes_grid) * len(rates_grid)
    output_path = Path("data/latency/localhost")
    clean_dir_local(output_path, dry_run=dry_run)
    for b in bytes_grid:

        for r in rates_grid:
            print(
                f"\n[{counter}/{total_experiments}]: \tHERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={r}..."
            )

            if b * r * flush_period_s > memory_limit:
                print(
                    f"[{counter}/{total_experiments}]: HERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={r} exceeds {memory_limit} memory limit."
                )
                counter += 1
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

            # TODO: verify dropout and time-dependent latency increase calculation
            # calc_cmd = [
            #     sys.executable,
            #     "utils/calc_latency.py",
            #     str(output_path),
            #     str(counter),
            #     str(r),
            #     str(b),
            #     "0",
            # ]
            # try:
            #     run_command(calc_cmd, dry_run=dry_run)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error calculating latency: {e}")
            #     if not args.continue_on_error:
            #         sys.exit(1)

            # clean_dir_local(
            #     output_path / "run_latency_vs_msgsize" / f"trial_{counter}",
            #     dry_run=dry_run,
            # )
            counter += 1

        # clean_dir_local(output_path / "run_latency_vs_msgsize", dry_run=dry_run)


def handle_multi_device(args):
    hermes_cli = get_hermes_cli_path()
    duration = args.duration
    dry_run = args.dry_run

    user = args.remote_user
    host = args.remote_host
    base_dir = args.remote_dir
    remote_os = args.remote_os

    # Pre-clean local directory structures
    clean_dir_local(
        Path("data/latency/multi_device/run_latency_vs_frequency"), dry_run=dry_run
    )
    clean_dir_local(
        Path("data/latency/multi_device/run_latency_vs_msgsize"), dry_run=dry_run
    )

    # Grid 1: latency vs frequency
    # We match the values from test_latency_multi_device.sh
    rates_grid_1 = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
    ]
    fixed_bytes_1 = 1000

    print("=== Starting Multi-Device Latency vs Frequency Sweep ===")
    counter = 0
    for r in rates_grid_1:
        print(
            f"\nStarting experiment {counter}: HERMES_EXP_NUM_BYTES={fixed_bytes_1}, HERMES_EXP_RATE={r}..."
        )

        env = os.environ.copy()
        env["HERMES_EXP_NUM_BYTES"] = str(fixed_bytes_1)
        env["HERMES_EXP_RATE"] = str(r)
        env["HERMES_EXP_BUF_LEN"] = str(r * 100)

        remote_path = f"{base_dir}/data/latency/multi_device/run_latency_vs_frequency/trial_{counter}"
        local_path = (
            Path("data/latency/multi_device/run_latency_vs_frequency")
            / f"trial_{counter}"
        )

        # Inject envs
        inject_cmd = [
            sys.executable,
            "utils/inject_envs.py",
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
            "data/latency/multi_device",
            "-d",
            str(duration),
            "--experiment",
            "run=latency_vs_frequency",
            f"trial={counter}",
            "-f",
            "../config/master.yml",
        ]
        try:
            run_command(cmd, env=env, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        # Copy results from remote device via SCP
        if not dry_run:
            local_path.mkdir(parents=True, exist_ok=True)

        scp_cmd = ["scp", f"{user}@{host}:{remote_path}/*", f"{local_path}/"]
        try:
            run_command(scp_cmd, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"SCP command failed: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        # Calculate latency
        calc_cmd = [
            sys.executable,
            "utils/calc_latency.py",
            "data/latency/multi_device",
            str(counter),
            str(r),
            str(fixed_bytes_1),
            "1",
        ]
        try:
            run_command(calc_cmd, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Error calculating latency: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        # Clean up local logs
        clean_dir_local(local_path, dry_run=dry_run)

        # Clean up remote logs via SSH
        try:
            clean_dir_remote(user, host, remote_path, remote_os, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Remote cleanup failed: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        print(
            f"Completed experiment {counter}: HERMES_EXP_NUM_BYTES={fixed_bytes_1}, HERMES_EXP_RATE={r}..."
        )
        counter += 1

    # Grid 2: latency vs msgsize
    bytes_grid_2 = [
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
        1000000,
    ]
    fixed_rate_2 = 100
    fixed_buf_len_2 = 10000

    print("\n=== Starting Multi-Device Latency vs Message Size Sweep ===")
    counter = 0
    for b in bytes_grid_2:
        print(
            f"\nStarting experiment {counter}: HERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={fixed_rate_2}..."
        )

        env = os.environ.copy()
        env["HERMES_EXP_NUM_BYTES"] = str(b)
        env["HERMES_EXP_RATE"] = str(fixed_rate_2)
        env["HERMES_EXP_BUF_LEN"] = str(fixed_buf_len_2)

        remote_path = f"{base_dir}/data/latency/multi_device/run_latency_vs_msgsize/trial_{counter}"
        local_path = (
            Path("data/latency/multi_device/run_latency_vs_msgsize")
            / f"trial_{counter}"
        )

        # Inject envs
        inject_cmd = [
            sys.executable,
            "utils/inject_envs.py",
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
            "data/latency/multi_device",
            "-d",
            str(duration),
            "--experiment",
            "run=latency_vs_msgsize",
            f"trial={counter}",
            "-f",
            "../config/master.yml",
        ]
        try:
            run_command(cmd, env=env, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        # Copy results from remote device via SCP
        if not dry_run:
            local_path.mkdir(parents=True, exist_ok=True)

        scp_cmd = ["scp", f"{user}@{host}:{remote_path}/*", f"{local_path}/"]
        try:
            run_command(scp_cmd, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"SCP command failed: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        # Calculate latency
        calc_cmd = [
            sys.executable,
            "utils/calc_latency.py",
            "data/latency/multi_device",
            str(counter),
            str(fixed_rate_2),
            str(b),
            "0",
        ]
        try:
            run_command(calc_cmd, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Error calculating latency: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        # Clean up local logs
        clean_dir_local(local_path, dry_run=dry_run)

        # Clean up remote logs via SSH
        try:
            clean_dir_remote(user, host, remote_path, remote_os, dry_run=dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Remote cleanup failed: {e}")
            if not args.continue_on_error:
                sys.exit(1)

        print(
            f"Completed experiment {counter}: HERMES_EXP_NUM_BYTES={b}, HERMES_EXP_RATE={fixed_rate_2}..."
        )
        counter += 1


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
        "--remote-user",
        type=str,
        default="a",
        help="SSH username for the remote slave device",
    )
    md_parser.add_argument(
        "--remote-host",
        type=str,
        default="10.220.25.100",
        help="SSH host/IP for the remote slave device",
    )
    md_parser.add_argument(
        "--remote-dir",
        type=str,
        default="C:/Users/a/Desktop/KDD2026/hermes",
        help="Base directory path on the remote slave device",
    )
    md_parser.add_argument(
        "--remote-os",
        type=str,
        choices=["windows", "linux"],
        default="windows",
        help="OS type of the remote slave device",
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
