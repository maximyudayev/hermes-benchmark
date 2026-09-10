import argparse
from pathlib import Path
import ffmpeg
import h5py
import matplotlib
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

matplotlib.rcParams['svg.fonttype'] = 'none'

def read_audio_ffmpeg(file_path: Path | str, sample_rate: int) -> np.ndarray:
    """Reads an audio file into a 1D float32 numpy array.

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate in Hz.

    Returns:
        1D float32 audio samples.
    """
    out, _ = (
        ffmpeg.input(str(file_path))
        .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
        .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, dtype=np.float32)


def get_audio_sample_rate(file_path: Path | str) -> int:
    """Queries the native sample rate of an audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        Native sample rate in Hz.
    """
    probe = ffmpeg.probe(str(file_path))
    audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    return int(audio_stream["sample_rate"])


def autocorrelate_audio(
    target_path: Path | str,
    operand_path: Path | str,
    threshold: float | None = None,
    min_distance: int | None = None,
    return_correlation: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Performs correlation/autocorrelation of target audio with operand audio.

    Finds the sample indices in the target WAV file where the operand sound is
    detected. Compatible with single or multiple occurrences of the sound.
    The complementary HDF5 file contains time-of-arrival of each chunk of N samples.

    Args:
        target_path: Path to the target audio file (e.g. mic_mic.wav).
        operand_path: Path to the operand audio file (e.g. cue.opus).
        threshold: Peak detection height threshold on the correlation values. If None,
            defaults to 0.35 * max(abs(correlation)) to reliably capture multiple
            instances of varying volume.
        min_distance: Minimum distance between detected peaks in samples. If None,
            defaults to len(operand).
        return_correlation: If True, returns a tuple of (detected_indices, correlation_array).

    Returns:
        Detected sample indices in the target audio where the operand sound was found,
        or a tuple of (detected_indices, correlation_array) if return_correlation is True.
    """
    sample_rate = get_audio_sample_rate(target_path)

    target = read_audio_ffmpeg(target_path, sample_rate=sample_rate)
    operand = read_audio_ffmpeg(operand_path, sample_rate=sample_rate)

    corr = signal.correlate(target, operand, mode="valid", method="fft")

    if threshold is None:
        threshold = 0.35 * float(np.max(np.abs(corr)))

    if min_distance is None:
        min_distance = max(1, len(operand))

    peaks, _ = signal.find_peaks(corr, height=threshold, distance=min_distance)

    if return_correlation:
        return peaks, corr
    return peaks


def plot_audio_correlation(
    corr: np.ndarray,
    sample_indices: np.ndarray,
    sample_rate: int,
    out_dir: str,
    threshold: float | None = None,
    title: str = "Audio Cross-Correlation",
) -> None:
    """Plots the cross-correlation signal over time with detected peaks highlighted.

    Args:
        corr: 1D correlation signal array.
        sample_indices: Array of sample indices corresponding to detected peaks.
        sample_rate: Audio sampling rate in Hz.
        threshold: Optional detection threshold line to display.
        title: Plot title.
    """
    time_s = np.arange(len(corr)) / sample_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_s, corr, label="Correlation", color="#1f77b4", linewidth=1.0)

    if threshold is not None:
        ax.axhline(
            threshold,
            color="#d62728",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({threshold:.2f})",
        )

    if len(sample_indices) > 0:
        ax.scatter(
            time_s[sample_indices],
            corr[sample_indices],
            color="#d62728",
            s=50,
            zorder=5,
            label=f"Detected ({len(sample_indices)})",
        )
        for idx in sample_indices:
            t = idx / sample_rate
            val = corr[idx]
            ax.annotate(
                f"{t:.2f}s",
                (t, val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correlation Amplitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(Path(out_dir) / "cross_correlation.svg")


def read_hdf5_dataset(filename: Path, dataset_name: str) -> np.ndarray:
    """Reads a dataset from an HDF5 file into a numpy array.

    Args:
        filename: Path to the HDF5 file.
        dataset_name: Name of the dataset within the HDF5 file.

    Returns:
        Dataset contents as a numpy array.
    """
    with h5py.File(filename, "r") as f:
        return np.array(f[dataset_name])


def interpolate_audio_toa(
    sample_indices: np.ndarray | list[int] | int,
    hdf5_path: Path | str,
    dataset_name: str = "mic/mic/toa_s",
    chunk_size: int = 960,
) -> np.ndarray | float:
    """Interpolates time-of-arrival (TOA) timestamps for exact audio sample indices.

    The HDF5 file contains a TOA timestamp for each chunk of N samples (default 960).
    Fits a linear function through the chunk timestamps across the whole recording to
    estimate the sample timestamps robustly against packet arrival jitter.

    Args:
        sample_indices: Sample index or array of sample indices.
        hdf5_path: Path to the HDF5 file containing the audio chunk TOA dataset.
        dataset_name: Dataset path within the HDF5 file. Defaults to 'mic/mic/toa_s'.
        chunk_size: Number of samples per chunk in the HDF5 file. Defaults to 960.

    Returns:
        Interpolated TOA timestamp(s) in seconds.
    """
    indices = np.atleast_1d(sample_indices)
    toa_s = read_hdf5_dataset(Path(hdf5_path), dataset_name).squeeze()

    chunk_sample_positions = (np.arange(len(toa_s)) + 1) * chunk_size - 1
    slope, intercept = np.polyfit(chunk_sample_positions, toa_s, 1)
    interpolated = slope * indices + intercept

    if np.isscalar(sample_indices):
        return float(interpolated.item())
    return interpolated


def plot_stacked_closed_loop_pipeline(
    imu_acc: np.ndarray,
    imu_toa_s: np.ndarray,
    ai_toa_s: np.ndarray,
    fog_detection_times: np.ndarray,
    cue_start_staged_times: np.ndarray,
    cue_stop_staged_times: np.ndarray,
    cue_start_sent_times: np.ndarray,
    cue_stop_sent_times: np.ndarray,
    buds_cue_ack_times: np.ndarray,
    mic_detected_times: np.ndarray,
    fog_label: np.ndarray,
    out_dir: str,
    zoom_cue: int,
    title: str = "Closed-Loop Auditory Cueing Pipeline",
) -> None:
    t0 = imu_toa_s[0]
    t_imu_s = imu_toa_s - t0
    t_imu_min = t_imu_s / 60.0
    t_ai_s = ai_toa_s - t0
    t_ai_min = t_ai_s / 60.0
    t_cue_start_s = cue_start_sent_times - t0
    t_cue_start_min = t_cue_start_s / 60.0
    t_cue_stop_s = cue_stop_sent_times - t0
    t_cue_stop_min = t_cue_stop_s / 60.0

    # Sensor indexing (based on IMU dataset heading definitions):
    # 0: pelvis, 1: knee_right, 2: foot_right, 3: knee_left, 4: foot_left
    acc_pelvis_x = imu_acc[:, 0, 0]
    acc_knee_r_x = imu_acc[:, 1, 0]
    acc_foot_r_x = imu_acc[:, 2, 0]
    acc_knee_l_x = imu_acc[:, 3, 0]
    acc_foot_l_x = imu_acc[:, 4, 0]

    # Find continuous intervals where fog_label == 1
    fog_flat = (fog_label == 1).flatten()
    diff = np.diff(fog_flat.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    fog_intervals_min = list(zip(t_imu_min[starts], t_imu_min[np.minimum(ends, len(t_imu_min) - 1)]))

    n_events = len(fog_intervals_min)

    # -------------------------------------------------------------
    # 1. PRIMARY FIGURE: Session Overview (Shared Horizontal Axis)
    # -------------------------------------------------------------
    fig_overview, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(14, 5.5),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.6], "hspace": 0},
    )

    # Row 1: Pelvis X Acceleration
    axes[0].plot(
        t_imu_min,
        acc_pelvis_x,
        label="Pelvis Accel X",
        color="#1f77b4",
        linewidth=0.8,
        zorder=2,
    )
    axes[0].set_ylabel("Pelvis Accel X", fontsize=10)
    axes[0].set_title(f"{title} - Overview", fontsize=13, pad=10)
    pelvis_ylim = axes[0].get_ylim()

    # Row 2: Knees X Acceleration (Left & Right overlaid)
    axes[1].plot(
        t_imu_min,
        acc_knee_l_x,
        label="Left Knee Accel X",
        color="#2ca02c",
        linewidth=0.8,
        alpha=0.75,
        zorder=2,
    )
    axes[1].plot(
        t_imu_min,
        acc_knee_r_x,
        label="Right Knee Accel X",
        color="#ff7f0e",
        linewidth=0.8,
        alpha=0.7,
        zorder=2,
    )
    axes[1].set_ylabel("Knees Accel X", fontsize=10)
    knees_ylim = axes[1].get_ylim()

    # Row 3: Feet X Acceleration (Left & Right overlaid)
    axes[2].plot(
        t_imu_min,
        acc_foot_l_x,
        label="Left Foot Accel X",
        color="#9467bd",
        linewidth=0.8,
        alpha=0.75,
        zorder=2,
    )
    axes[2].plot(
        t_imu_min,
        acc_foot_r_x,
        label="Right Foot Accel X",
        color="#d62728",
        linewidth=0.8,
        alpha=0.7,
        zorder=2,
    )
    axes[2].set_ylabel("Feet Accel X", fontsize=10)
    feet_ylim = axes[2].get_ylim()

    # Remove vertical ticks and tighten margins for the first 3 subplots
    for ax in axes[:3]:
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
        ax.margins(y=0)

    # Remove outline / spines for the subplots
    for ax in axes:
        ax.spines[:].set_visible(False)

    # Row 4: Cueing & AI Signal (Square wave starting at mic detected audio)
    x = np.column_stack((t_cue_start_min, t_cue_start_min, t_cue_stop_min, t_cue_stop_min)).ravel()
    y = np.tile([0, 1, 1, 0], len(t_cue_start_min))

    axes[3].plot(
        np.concatenate([t_imu_min[0:1], x, t_imu_min[-2:-1]], axis=0),
        np.concatenate([np.array([0]), y, np.array([0])], axis=0),
        color="#d62728",
        linewidth=1,
        zorder=2,
    )
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["OFF", "ON"])

    # Highlight FOG label ground truth as grey background beneath line plots (Rows 1-3)
    for t_s, t_e in fog_intervals_min:
        for ax in axes[:3]:
            ax.axvspan(t_s, t_e, facecolor="#808080", alpha=0.25, zorder=0)

    # Format horizontal time axis in minutes
    axes[-1].set_xlabel("Entire recording (min)", fontsize=10)
    axes[-1].xaxis.set_major_locator(MultipleLocator(2))

    fig_overview.savefig(Path(out_dir) / "cue_latency_overview.svg", bbox_inches="tight")

    # -------------------------------------------------------------
    # 2. SECONDARY FIGURE: Zoomed Segment of Interest & Latency Breakdown
    # -------------------------------------------------------------
    fog_intervals_s = list(zip(t_imu_s[starts], t_imu_s[np.minimum(ends, len(t_imu_s) - 1)]))
    for i in range(n_events):
        t_gt_start_s, t_gt_end_s = fog_intervals_s[i]

        t_fog_detected_s = (fog_detection_times[i] - t0)
        t_cue_staged_s = (cue_start_staged_times[i] - t0)
        t_cue_unstaged_s = (cue_stop_staged_times[i] - t0)
        t_cue_start_s = (cue_start_sent_times[i] - t0)
        t_cue_stop_s = (cue_stop_sent_times[i] - t0)
        t_mic_detected_s = (mic_detected_times[i] - t0)

        # Center zoom window around the cue event: from slightly before AI FOG detection
        #   to past the cue pulse completion
        z_start_s = max(0.0, t_gt_start_s - 0.1)
        z_end_s = min(t_imu_s[-1], t_mic_detected_s + 0.1)

        imu_mask = (t_imu_s >= z_start_s) & (t_imu_s <= z_end_s)

        fig_zoom, z_axes = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=(5, 5.5),
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.6], "hspace": 0},
        )

        # Row 1: Pelvis X Accel (zoomed)
        z_axes[0].plot(
            (t_imu_s[imu_mask]-t_gt_start_s) * 1000.0,
            acc_pelvis_x[imu_mask],
            label="Pelvis Accel X",
            color="#1f77b4",
            linewidth=0.8,
            zorder=2,
        )
        z_axes[0].set_title(f"{title} - Zoomed", fontsize=13, pad=10)
        z_axes[0].set_ylim(*pelvis_ylim)

        # Row 2: Knees X Accel (zoomed)
        z_axes[1].plot(
            (t_imu_s[imu_mask]-t_gt_start_s) * 1000.0,
            acc_knee_l_x[imu_mask],
            label="Left Knee Accel X",
            color="#2ca02c",
            linewidth=0.8,
            alpha=0.75,
            zorder=2,
        )
        z_axes[1].plot(
            (t_imu_s[imu_mask]-t_gt_start_s) * 1000.0,
            acc_knee_r_x[imu_mask],
            label="Right Knee Accel X",
            color="#ff7f0e",
            linewidth=0.8,
            alpha=0.7,
            zorder=2,
        )
        z_axes[1].set_ylim(*knees_ylim)

        # Row 3: Feet X Accel (zoomed)
        z_axes[2].plot(
            (t_imu_s[imu_mask]-t_gt_start_s) * 1000.0,
            acc_foot_l_x[imu_mask],
            label="Left Foot Accel X",
            color="#9467bd",
            linewidth=0.8,
            alpha=0.75,
            zorder=2,
        )
        z_axes[2].plot(
            (t_imu_s[imu_mask]-t_gt_start_s) * 1000.0,
            acc_foot_r_x[imu_mask],
            label="Right Foot Accel X",
            color="#d62728",
            linewidth=0.8,
            alpha=0.7,
            zorder=2,
        )
        z_axes[2].set_ylim(*feet_ylim)

        # Remove vertical ticks and tighten margins for the first 3 subplots
        for ax in z_axes[:3]:
            ax.set_yticks([])
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
            ax.margins(y=0)

        # Remove outline / spines for the subplots
        for ax in z_axes:
            ax.spines[:].set_visible(False)

        # Row 4: Cue & AI square waves (zoomed)
        # if z_end_s < t_cue_unstaged_s:
        z_ai_x = np.array([z_start_s, t_fog_detected_s, t_fog_detected_s, z_end_s])
        z_staged_x = np.array([z_start_s, t_cue_staged_s, t_cue_staged_s, z_end_s])
        z_sent_x = np.array([z_start_s, t_cue_start_s, t_cue_start_s, z_end_s])
        z_mic_x = np.array([z_start_s, t_mic_detected_s, t_mic_detected_s, z_end_s])
        z_y = np.array([0, 0, 1, 1])
        # else:
        #     z_ai_x = np.array([z_start_s, t_fog_detected_s, t_fog_detected_s, z_end_s])
        #     z_staged_x = np.array([z_start_s, t_cue_staged_s, t_cue_staged_s, t_cue_unstaged_s, t_cue_unstaged_s, z_end_s])
        #     z_sent_x = np.array([z_start_s, t_cue_start_s, t_cue_start_s, t_cue_stop_s, t_cue_stop_s, z_end_s])
        #     z_mic_x = np.array([z_start_s, t_mic_detected_s, t_mic_detected_s, t_cue_unstaged_s, t_cue_unstaged_s, z_end_s])
        #     z_y = np.array([0, 0, 1, 1, 0, 0])

        z_axes[3].plot(
            (z_mic_x-t_gt_start_s) * 1000.0,
            z_y,
            color="#d62728",
            alpha=1.0,
            linewidth=0.8,
            zorder=2,
        )
        z_axes[3].plot(
            (z_sent_x-t_gt_start_s) * 1000.0,
            z_y,
            color="#d62728",
            alpha=0.75,
            linewidth=0.8,
            zorder=2,
        )
        z_axes[3].plot(
            (z_staged_x-t_gt_start_s) * 1000.0,
            z_y,
            color="#d62728",
            alpha=0.50,
            linewidth=0.8,
            zorder=2,
        )
        z_axes[3].plot(
            (z_ai_x-t_gt_start_s) * 1000.0,
            z_y,
            color="#d62728",
            alpha=0.25,
            linewidth=0.8,
            zorder=2,
        )
        z_axes[3].set_ylim(-0.1, 1.1)
        z_axes[3].set_yticks([0, 1])
        z_axes[3].set_yticklabels(["OFF", "ON"])

        # Highlight FOG label ground truth in grey beneath lines
        for t_s, t_e in fog_intervals_s:
            if not (t_e < z_start_s or t_s > z_end_s):
                for ax in z_axes[:3]:
                    ax.axvspan(
                        (max(t_s, z_start_s) - t_gt_start_s) * 1000.0,
                        (min(t_e, z_end_s) - t_gt_start_s) * 1000.0,
                        facecolor="#808080",
                        alpha=0.25,
                        zorder=0
                    )

        # Format horizontal time axis in minutes (NOTE: centered at 0ms)
        z_axes[-1].set_xlabel("Time around cue (ms)", fontsize=11)

        fig_zoom.savefig(Path(out_dir) / f"cue_latency_zoomed_{i}.svg")


def load_dataset(file_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    # Load IMU data: raw shape (N_sensors_raw, 6, n_samples)
    npy_data = np.load(
        file_path, allow_pickle=True
    ).item()  # Load the dictionary from the NPY file
    labels = npy_data["motor_labels"]  # (n_samples,)

    imu_acc = npy_data["imu_acc"]  # (n_samples, n_sensors, 3)

    # Concatenate all trials along the time axis
    num_frames = imu_acc.shape[2]
    return imu_acc, labels, num_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect occurrences of an operand sound in a target audio file via cross-correlation."
    )
    parser.add_argument(
        "target_audio",
        type=str,
        help="Path to the target audio file (e.g. mic.wav).",
    )
    parser.add_argument(
        "operand_audio",
        type=str,
        help="Path to the operand audio file (e.g. cue.opus).",
    )
    parser.add_argument(
        "mic_hdf5",
        type=str,
        help="Path to the complementary HDF5 file containing audio chunk arrival time.",
    )
    parser.add_argument(
        "imu_hdf5",
        type=str,
        help="Path to the IMU HDF5.",
    )
    parser.add_argument(
        "ai_hdf5",
        type=str,
        help="Path to AI outputs HDF5.",
    )
    parser.add_argument(
        "buds_hdf5",
        type=str,
        help="Path to earbuds HDF5.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to serialized NPY dataset file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=960,
        help="Number of samples per chunk in the HDF5 file (default: 960).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Peak detection height threshold (default: 0.35 * max correlation).",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="Minimum distance between detected peaks in samples (default: len(operand)).",
    )
    parser.add_argument(
        "--cue-duration",
        type=float,
        default=1.0,
        help="Duration of the audio cue square wave pulse in seconds (default: 1.0s).",
    )
    parser.add_argument(
        "--zoom-cue",
        type=int,
        default=0,
        help="Zoom into a specific cue event window for the secondary figure (default: 0).",
    )

    args = parser.parse_args()

    sample_indices, corr = autocorrelate_audio(
        args.target_audio,
        args.operand_audio,
        threshold=args.threshold,
        min_distance=args.min_distance,
        return_correlation=True,
    )

    sample_rate = get_audio_sample_rate(args.target_audio)

    mic_path = Path(args.mic_hdf5)
    mic_detected_times = np.atleast_1d(
        interpolate_audio_toa(
            sample_indices,
            mic_path,
            chunk_size=args.chunk_size,
        )
    )

    first_toa = read_hdf5_dataset(mic_path, "mic/mic/toa_s")[0, 0]

    # threshold_val = (
    #     args.threshold
    #     if args.threshold is not None
    #     else 0.35 * float(np.max(np.abs(corr)))
    # )

    # plot_title = (
    #     f"Cross-Correlation: {Path(args.target_audio).name} vs {Path(args.operand_audio).name}"
    # )
    
    # plot_audio_correlation(
    #     corr,
    #     sample_indices,
    #     sample_rate,
    #     threshold=threshold_val,
    #     title=plot_title,
    #     out_dir=args.out_dir,
    # )

    # Load data timestamps.
    imu_path = Path(args.imu_hdf5)
    ai_path = Path(args.ai_hdf5)
    buds_path = Path(args.buds_hdf5)

    # acc, labels, num_frames = load_dataset(args.dataset)

    with h5py.File(imu_path, "r") as f_imu:
        imu_acc = np.array(f_imu["imu_replay/dots_imu/acceleration"])
        imu_toa_s = np.array(f_imu["imu_replay/dots_imu/toa_s"])[:, 0]
        fog_label = np.array(f_imu["imu_replay/dots_imu/fog_label"])[:, 0]

    with h5py.File(ai_path, "r") as f_ai:
        ai_fog = np.array(f_ai["ai_fog/pytorch_worker/binary"])[:, 0]
        ai_toa_s = np.array(f_ai["ai_fog/pytorch_worker/toa_s"])[:, 0]
        diff = np.diff(ai_fog, prepend=0)
        fog_detection_times = ai_toa_s[diff == 1]

    with h5py.File(buds_path, "r") as f_buds:
        # When the command was staged for transmission by the FSM and when it was written over BLE.
        buds_events = np.array(f_buds["buds/buds_event/event"])[:, 0]
        buds_cmd_sent_s = np.array(f_buds["buds/buds_event/cmd_sent_s"])[:, 0]
        buds_event_toa_s = np.array(f_buds["buds/buds_event/toa_s"])[:, 0]
        cue_start_staged_times = buds_cmd_sent_s[buds_events == 1]
        cue_stop_staged_times = buds_cmd_sent_s[buds_events == 2][:-1]
        cue_start_sent_times = buds_event_toa_s[buds_events == 1]
        cue_stop_sent_times = buds_event_toa_s[buds_events == 2][:-1]

        # When the buds acknowledged the new status.
        buds_cue = np.array(f_buds["buds/cueing_status/status"])[:, 0]
        buds_cue_toa_s = np.array(f_buds["buds/cueing_status/toa_s"])[:, 0]
        buds_cue_ack_times = buds_cue_toa_s[buds_cue == 1]

    # Print detailed latency pipeline breakdown table
    # print("\n=== Closed-Loop Cueing Latency Pipeline ===")
    # output_path = Path(args.out_dir)
    # with open(output_path / "cue_latency.csv", "w") as f:
    #     f.write("episode,ai_detected,cue_triggered,ble_cmd_sent,ble_cmd_acked,mic_detected,total\n")
        
    #     for i in range(len(mic_detected_times)):
    #         t_fog_detected = fog_detection_times[i]
    #         t_cue_triggered = cue_cmd_staged_times[i]
    #         t_ble_cmd_sent = cue_cmd_sent_times[i]
    #         t_ble_cmd_acked = buds_cue_ack_times[i]
    #         t_mic_detected = mic_detected_times[i]

    #         print(f"Event {i+1}:")
    #         print(f"  1. AI FOG detected:               {t_fog_detected:.6f}s (t={t_fog_detected - first_toa:8.3f}s)")
    #         print(f"  2. Earbuds FSM cue triggered:     {t_cue_triggered:.6f}s (+{(t_cue_triggered - t_fog_detected)*1e3:6.1f} ms)")
    #         print(f"  3. BLE cue command sent:          {t_ble_cmd_sent:.6f}s (+{(t_ble_cmd_sent - t_cue_triggered)*1e3:6.1f} ms)")
    #         print(f"  4. Cue acknowledged by buds:      {t_ble_cmd_acked:.6f}s (+{(t_ble_cmd_acked - t_ble_cmd_sent)*1e3:6.1f} ms)")
    #         print(f"  5. Mic detected audio playback:   {t_mic_detected:.6f}s (+{(t_mic_detected - t_ble_cmd_acked)*1e3:6.1f} ms)")
    #         print(f"  Total closed-loop latency:        {(t_mic_detected - t_fog_detected)*1e3:6.1f} ms\n")

    #         f.write(
    #             f"{i},"
    #             f"{t_fog_detected},"
    #             f"{t_cue_triggered},"
    #             f"{t_ble_cmd_sent},"
    #             f"{t_ble_cmd_acked},"
    #             f"{t_mic_detected},"
    #             f"{t_mic_detected - t_fog_detected}\n"
    #         )

    # TODO: create a sequence of horizontal whiskered bar charts visualizing the latency of each stage.


    plot_stacked_closed_loop_pipeline(
        imu_acc=imu_acc,
        imu_toa_s=imu_toa_s,
        ai_toa_s=ai_toa_s,
        fog_detection_times=fog_detection_times,
        cue_start_staged_times=cue_start_staged_times,
        cue_stop_staged_times=cue_stop_staged_times,
        cue_start_sent_times=cue_start_sent_times,
        cue_stop_sent_times=cue_stop_sent_times,
        buds_cue_ack_times=buds_cue_ack_times,
        mic_detected_times=mic_detected_times,
        fog_label=fog_label,
        zoom_cue=args.zoom_cue,
        out_dir=args.out_dir,
    )
