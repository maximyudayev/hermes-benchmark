import argparse
from pathlib import Path
import pandas as pd
from scipy import stats
import sys


def analyze_jitter_trends(stats_target: Path):
    """
    Reads the aggregated jitter statistics and performs a one-sample t-test
    on the jitter slope to check for temporal trends across trials.

    Args:
        stats_target (Path): Path to a 'stats.csv' file or a directory containing 'stats.csv'
                             (either directly or in trial subdirectories).
    """
    if not stats_target.exists():
        print(f"Error: Statistics path not found at '{stats_target}'")
        sys.exit(1)

    # Collect stats.csv file(s)
    if stats_target.is_file():
        stats_files = [stats_target]
    elif (stats_target / "stats.csv").is_file():
        stats_files = [stats_target / "stats.csv"]
    else:
        stats_files = sorted(list(stats_target.rglob("stats.csv")))

    if not stats_files:
        print(f"Error: No 'stats.csv' file found in '{stats_target}' or its subdirectories.")
        sys.exit(1)

    dfs = []
    for f in stats_files:
        try:
            cur_df = pd.read_csv(f)
            dfs.append(cur_df)
        except Exception as e:
            print(f"Warning: Could not read '{f}': {e}")

    if not dfs:
        print("Error: No valid stats data loaded.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)

    if "jitter_slope" not in df.columns:
        print("Error: 'jitter_slope' column not found in the stats file.")
        print(
            "Please re-run 'gen_jitter_distribution.py' or 'evaluate_jitter.py' to generate this column."
        )
        sys.exit(1)

    # Determine grouping based on available columns
    if "group" in df.columns and "name" in df.columns:
        grouped = df.groupby(["group", "name"])
    elif "name" in df.columns:
        grouped = df.groupby(["name"])
    else:
        df["_modality"] = "All Modalities"
        grouped = df.groupby(["_modality"])

    results = []
    for keys, data in grouped:
        if isinstance(keys, tuple):
            modality_name = f"{keys[0]}/{keys[1]}"
        else:
            modality_name = str(keys)

        slopes = data["jitter_slope"].dropna()
        num_trials = len(slopes)

        if num_trials < 2:
            single_slope = slopes.iloc[0] if num_trials == 1 else 0.0
            print(
                f"Notice [{modality_name}]: Found {num_trials} trial (slope: {single_slope:.4e} ms/sample). "
                f"At least 2 trials are required for statistical hypothesis testing (t-test)."
            )
            continue

        # Check for zero or near-zero variance
        std_val = slopes.std()
        if std_val == 0 or pd.isna(std_val):
            t_stat, p_value = 0.0, 1.0
        else:
            # H₀: The true mean slope is >= 0 (jitter compounds or is constant)
            # H₁: The true mean slope is < 0 (jitter decreases or does not compound over time)
            t_stat, p_value = stats.ttest_1samp(slopes, 0, alternative="less")

        results.append(
            {
                "Modality": modality_name,
                "Trials": num_trials,
                "Mean Slope (ms/sample)": f"{slopes.mean():.4e}",
                "Std Slope (ms/sample)": f"{std_val:.4e}",
                "t-statistic": f"{t_stat:.4f}",
                "p-value": f"{p_value:.4f}",
            }
        )

    if not results:
        print("\nNo modalities with >= 2 trials available for statistical trend hypothesis testing.")
        return

    # Print results in a formatted table
    results_df = pd.DataFrame(results)
    print("\n--- Jitter Trend Analysis ---")
    print("Hypothesis Test: Testing if jitter slope is significantly less than zero (one-sample t-test).")
    print(
        "A small p-value (< 0.05) rejects H0 and suggests jitter does not compound over time.\n"
    )
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform statistical analysis on jitter trends from multiple trials.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        required=True,
        help="Path to the directory containing 'stats.csv' or multiple trial subdirectories with 'stats.csv'.",
    )

    args = parser.parse_args()
    analyze_jitter_trends(Path(args.dir))
