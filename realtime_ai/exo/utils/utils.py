import os
import re
import ffmpeg
import h5py
import matplotlib.patches as mpatches
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.metrics as metrics
import torch
from torch.nn import Module

from ..models import get_model_class
from .types import Config, ModelType

matplotlib.rcParams['svg.fonttype'] = 'none'

IMU_JOINT_NAMES = [
    "Pelvis",
    "Right_Upper_Leg",
    "Right_Lower_Leg",
    "Right_Foot",
    "Left_Upper_Leg",
    "Left_Lower_Leg",
    "Left_Foot",
]

LABEL_COLORS = {
    "Level ground walking": "#3F90DA",  # Distinct Academic Blue
    "Sit to stand": "#FFA90E",  # Bright Amber/Orange
    "Stand to sit": "#BD1F01",  # Deep Red
    "Sitting": "#832DB6",  # Rich Purple
    "Stair up": "#32A251",  # Solid Green
    "Stair down": "#B3E188",  # Soft Green
    "Ramp up": "#FF95A8",  # Soft Pink
    "Ramp down": "#C44E52",  # Muted Crimson
    "Grass walking": "#D0B883",  # Khaki/Tan
    "Uneven ground walking": "#8D6F64",  # Earthy Brown
    "Carry": "#5F9ED1",  # Soft Sky Blue
    "Unknown": "#EAEAEA",  # Light Neutral Grey
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],  # A common font for papers
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.markersize": 5,
    }
)


def _hex_to_rgb(h):
    return (int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255)


def _labels_to_colors(labels):
    return np.array([_hex_to_rgb(LABEL_COLORS[l]) for l in labels])


def _draw_ribbon(ax, time, labels):
    colors = _labels_to_colors(labels)
    ribbon = colors[np.newaxis, :, :]
    ax.imshow(
        ribbon,
        aspect="auto",
        interpolation="nearest",
        extent=[time.iloc[0], time.iloc[-1], 0, 1],
    )
    ax.set_xlim(time.iloc[0], time.iloc[-1])
    ax.set_yticks([])
    ax.tick_params(bottom=False, labelbottom=False)


def _legend_handles(present):
    return [
        mpatches.Patch(color=LABEL_COLORS[l], label=l)
        for l in LABEL_COLORS
        if l in present
    ]


def fill_annotations_from_csv(df: pd.DataFrame, annoations_path: str) -> pd.DataFrame:
    annotation_df = pd.read_csv(annoations_path)
    for id, row in annotation_df.iterrows():
        # TODO: select whether annotation is added w.r.t. inference time, or the underlying window.
        mask = (df["toa_s"] >= row["window_start"]) & (
            df["toa_s"] <= row["window_end"]
        )
        df.loc[mask, "true_class"] = row["true_class"]
        df.loc[mask, "true_idx"] = row["true_idx"]
    return df


def save_inference_results(
    df: pd.DataFrame, output_dir: str, prediction_horizons: list[float]
):
    # NOTE: window start/end are NumPy arrays and get malformed when written from Pandas to CSV
    df.to_csv(os.path.join(output_dir, "predictions_all_horizons.csv"), index=False)
    for h in prediction_horizons:
        h_df = df[df["horizon_s"] == h]
        h_str = f"{h:.1f}".replace(".", "_")
        h_df.to_csv(
            os.path.join(output_dir, f"predictions_horizon_{h_str}s.csv"),
            index=False,
        )


def plot_all_horizons(df, save_path, title=""):
    horizons = sorted(df["horizon_s"].unique())
    n_rows = 1 + len(horizons)
    fig = plt.figure(figsize=(18, 0.65 * n_rows + 0.8))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, 0.18], wspace=0.02, hspace=0.08)

    gt_df = (
        df[df["horizon_s"] == horizons[0]]
        .sort_values("window_end_relative_s")
        .reset_index(drop=True)
    )
    gt_time = gt_df["window_end_relative_s"]
    t_start = gt_time.iloc[0]
    t_end = gt_time.iloc[-1] + max(horizons)
    present = set(gt_df["true_class"].unique())

    ax_gt = fig.add_subplot(gs[0, 0])
    _draw_ribbon(ax_gt, gt_time, gt_df["true_class"])
    ax_gt.set_xlim(t_start, t_end)
    ax_gt.set_ylabel("GT", fontsize=12, rotation=0, labelpad=25, va="center")

    last_ax = ax_gt
    for h_idx, h in enumerate(horizons):
        h_df = (
            df[df["horizon_s"] == h]
            .sort_values("window_end_relative_s")
            .reset_index(drop=True)
        )
        shifted_time = h_df["window_end_relative_s"] + h
        present |= set(h_df["predicted_class"].unique())
        ax = fig.add_subplot(gs[1 + h_idx, 0], sharex=ax_gt)
        _draw_ribbon(ax, shifted_time, h_df["predicted_class"])
        ax.set_ylabel(f"+{h}s", fontsize=12, rotation=0, labelpad=20, va="center")
        if h_idx < len(horizons) - 1:
            ax.tick_params(bottom=False, labelbottom=False)
        last_ax = ax
    last_ax.tick_params(bottom=True, labelbottom=True)
    last_ax.set_xlabel("Time (s)", fontsize=12)

    if title:
        ax_gt.set_title(title, fontsize=14, loc="left", pad=4)

    ax_legend = fig.add_subplot(gs[:, 1])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=_legend_handles(present),
        loc="center left",
        fontsize=10,
        frameon=False,
        handlelength=1.2,
        handleheight=1.0,
        labelspacing=0.4,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _majority_vote(labels, window):
    out = list(labels)
    for i in range(len(labels)):
        lo = max(0, i - window + 1)
        out[i] = Counter(labels[lo : i + 1]).most_common(1)[0][0]
    return out


def smooth_predictions(df: pd.DataFrame, window: int) -> pd.DataFrame:
    parts = []
    for h, grp in df.groupby("horizon_s", sort=False):
        grp = grp.sort_values("window_end_relative_s").copy()
        grp["predicted_class"] = _majority_vote(
            grp["predicted_class"].tolist(), window=window
        )
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def plot_with_smoothed(df, output_dir, title="", smooth_window=5):
    plot_all_horizons(df, os.path.join(output_dir, "all_horizons.svg"), title=title)
    plot_all_horizons(
        smooth_predictions(df, window=smooth_window),
        os.path.join(output_dir, "all_horizons_smooth.svg"),
        title=f"{title} (smooth={smooth_window})"
        if title
        else f"(smooth={smooth_window})",
    )


def build_imu_column_names():
    cols = []
    for joint in IMU_JOINT_NAMES:
        for axis in ["X", "Y", "Z"]:
            cols.append(f"acc_{joint}_{axis}")
    for joint in IMU_JOINT_NAMES:
        for axis in ["X", "Y", "Z"]:
            cols.append(f"gyro_{joint}_{axis}")
    return cols


def select_imu_columns(imu_array, column_patterns):
    all_cols = build_imu_column_names()
    selected = []
    for pattern in column_patterns:
        regex = re.compile(pattern.replace("*", ".*"))
        for i, c in enumerate(all_cols):
            if regex.fullmatch(c) and i not in selected:
                selected.append(i)
    return imu_array[:, selected]


def load_imu_data(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        root = "mvn-analyze" if "mvn-analyze" in f else "mvn"
        mt = f[f"{root}/xsens-motion-trackers"]
        acc = mt["acceleration"][:]
        gyro = mt["gyroscope"][:]
        timestamps = mt["toa_s"][:, 0]

    N = len(timestamps)
    imu_array = np.concatenate(
        [
            acc.reshape(N, -1),
            gyro.reshape(N, -1),
        ],
        axis=1,
    ).astype(np.float32)
    return imu_array, timestamps


def load_ai_metadata(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        ai = f["ai_intent/classifier"]
        ai_window_starts = ai["window_start_s"][:]
        ai_window_ends = ai["window_end_s"][:]
        ai_timestamps = ai["toa_s"][:, 0]

    return ai_window_starts, ai_window_ends, ai_timestamps


def load_video_metadata(hdf5_path, video_path):
    with h5py.File(hdf5_path, "r") as f:
        ego = f["smartglasses/ego"]
        timestamps = ego["toa_s"][:, 0]

    probe = ffmpeg.probe(video_path)
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps_num, fps_denum = map(
        lambda x: float(x), video_stream["r_frame_rate"].split("/")
    )
    fps = fps_num / fps_denum
    num_frames = round(float(probe["format"]["duration"]) * fps)

    return timestamps, width, height, fps, num_frames


def load_video_data(
    start_frame: int,
    end_frame: int,
    fps: float,
    height: int,
    width: int,
    video_path: str,
):
    # Initialize cache on the function if it doesn't exist
    if not hasattr(load_video_data, "cache"):
        load_video_data.cache = {
            "start_frame": -1,
            "end_frame": -1,
            "video_array": None,
            "video_path": None,
        }

    c = load_video_data.cache

    # Load a new chunk if the cache is invalid or doesn't cover the requested range
    if (
        c["video_array"] is None
        or c["video_path"] != video_path
        or start_frame < c["start_frame"]
        or end_frame > c["end_frame"]
    ):
        # Cache a 300 frame block by default to avoid excessive ffmpeg calls
        chunk_frames = max(300, end_frame - start_frame + 1)
        c["start_frame"] = start_frame
        c["end_frame"] = start_frame + chunk_frames - 1
        c["video_path"] = video_path

        # Seek to the timestamp because it is much faster than using frame index
        timestamp_start = c["start_frame"] / fps
        timestamp_end = (c["end_frame"] + 1) / fps
        duration = timestamp_end - timestamp_start

        video_buf, _ = (
            ffmpeg.input(filename=video_path, ss=timestamp_start)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                t=duration,
            )
            .run(capture_stdout=True, quiet=True)
        )

        if video_buf:
            c["video_array"] = (
                torch.frombuffer(video_buf, dtype=torch.uint8)
                .reshape(-1, height, width, 3)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            c["video_array"] = torch.zeros((1, 3, height, width), dtype=torch.uint8)

    # Slice relative to the cached chunk
    rel_start = start_frame - c["start_frame"]
    rel_end = end_frame - c["start_frame"]

    return c["video_array"][rel_start : rel_end + 1]  # [T, C, H, W]


def load_ai_predictions(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        ai = f["ai_intent/classifier"]
        preds = ai["predictions"][:]
        window_starts = ai["window_start_s"][:]
        window_ends = ai["window_end_s"][:]
        timestamps = ai["toa_s"][:, 0]
        compute_times = ai["compute_time_s"][:, 0]
    return preds, window_starts, window_ends, timestamps, compute_times


def build_model(config: Config, device: str | torch.device) -> Module:
    """Build inference-only model from configuration

    Args:
        config (Config): Top-level configuration of the live AI model.
        device (str | torch.device): Device on which to run the AI inference of the built model.

    Returns:
        Module: Model in eval mode, built from the config, and loaded from the checkpoint.
    """

    models: dict[str, Module] = {}
    for model_type, model_config in config.models.items():
        if model_type == ModelType.FUSION:
            continue
        cls = get_model_class(model_config.name)
        params = model_config.params.copy()
        params.setdefault("num_classes", config.num_classes)
        params.setdefault("prediction_horizons", config.prediction_horizons)
        models[model_type.value] = cls(**params).to(device)

    if ModelType.FUSION in config.models:
        fusion_config = config.models[ModelType.FUSION]
        params = fusion_config.params.copy()
        params.setdefault("num_classes", config.num_classes)
        params.setdefault("prediction_horizons", config.prediction_horizons)
        encoders: dict[str, Module] = {}
        for mod in config.modalities.keys():
            if mod.value in models:
                encoders[mod.value] = models[mod.value]
        params["modality_encoders"] = encoders
        main_model: Module = get_model_class(fusion_config.name)(**params)
        main_model = main_model.to(device)
    else:
        main_model = next(
            models[k]
            for k in [
                ModelType.VIDEO.value,
                ModelType.IMAGE.value,
                ModelType.RAW_IMU.value,
            ]
            if k in models
        )

    # Load checkpoint.
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        ckpt: dict = torch.load(
            config.checkpoint_path, map_location=device, weights_only=False
        )
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        try:
            main_model.load_state_dict(sd, strict=True)
            print("Checkpoint loaded (strict)", flush=True)
        except RuntimeError:
            missing, unexpected = main_model.load_state_dict(sd, strict=False)
            print(
                f"Checkpoint loaded (non-strict): {len(missing)} missing, {len(unexpected)} unexpected keys",
                flush=True,
            )
    elif config.checkpoint_path:
        print(f"Warning: Checkpoint not found: {config.checkpoint_path}", flush=True)

    # TODO: compile the model with Dynamo for higher efficiency.

    return main_model.eval()


def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sample-wise and segmental KPIs for temporal continuous predictions.
    Computes: Accuracy, Macro F1, F1@{10, 25, 50}, Levenshtein Distance, and Detection Latency.
    """
    results = []
    
    if "horizon_s" in df.columns:
        horizons = sorted(df["horizon_s"].unique())
    else:
        horizons = [0.0]
        
    for h in horizons:
        if "horizon_s" in df.columns:
            h_df = df[df["horizon_s"] == h].sort_values("window_end_relative_s").reset_index(drop=True)
        else:
            h_df = df.sort_values("window_end_relative_s").reset_index(drop=True)
            
        valid_df = h_df[h_df["true_class"] != "Unknown"]
        if valid_df.empty:
            continue
            
        y_true = valid_df["true_class"].tolist()
        y_pred = valid_df["predicted_class"].tolist()
        
        acc = metrics.accuracy_score(y_true, y_pred)
        macro_f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Segmental metrics
        times = h_df["window_end_relative_s"].tolist()
        all_true = h_df["true_class"].tolist()
        all_pred = h_df["predicted_class"].tolist()
        
        def get_temporal_segments(classes, times):
            segments = []
            if len(classes) == 0:
                return segments
            curr_c = classes[0]
            start_t = times[0]
            last_t = times[0]
            for c, t in zip(classes[1:], times[1:]):
                if c != curr_c:
                    if curr_c != "Unknown":
                        segments.append({'class': curr_c, 'start': start_t, 'end': last_t})
                    curr_c = c
                    start_t = t
                last_t = t
            if curr_c != "Unknown":
                segments.append({'class': curr_c, 'start': start_t, 'end': last_t})
            return segments
            
        true_segments = get_temporal_segments(all_true, times)
        pred_segments = get_temporal_segments(all_pred, times)
        
        def edit_distance(seq1, seq2):
            n, m = len(seq1), len(seq2)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(n + 1): dp[i][0] = i
            for j in range(m + 1): dp[0][j] = j
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
            return dp[n][m]
            
        true_seq = [s['class'] for s in true_segments]
        pred_seq = [s['class'] for s in pred_segments]
        lev_dist = edit_distance(true_seq, pred_seq)
        
        def calculate_iou(seg1, seg2):
            start = max(seg1['start'], seg2['start'])
            end = min(seg1['end'], seg2['end'])
            intersection = max(0, end - start)
            union = max(seg1['end'], seg2['end']) - min(seg1['start'], seg2['start'])
            if union <= 0:
                return 0.0
            return intersection / union
            
        def calculate_f1_at_k(t_segs, p_segs, k):
            tp = 0
            matched_true = set()
            matched_pred = set()
            matches = []
            for i, t_seg in enumerate(t_segs):
                for j, p_seg in enumerate(p_segs):
                    if t_seg['class'] == p_seg['class']:
                        iou = calculate_iou(t_seg, p_seg)
                        if iou >= k:
                            matches.append((iou, i, j))
            for iou, i, j in matches:
                if i not in matched_true and j not in matched_pred:
                    tp += 1
                    matched_true.add(i)
                    matched_pred.add(j)
            fp = len(p_segs) - tp
            fn = len(t_segs) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return f1

        f1_10 = calculate_f1_at_k(true_segments, pred_segments, 0.1)
        f1_25 = calculate_f1_at_k(true_segments, pred_segments, 0.25)
        f1_50 = calculate_f1_at_k(true_segments, pred_segments, 0.5)

        # Detection Latency
        latencies = []
        for t_seg in true_segments:
            mask = (h_df['window_end_relative_s'] >= t_seg['start']) & (h_df['window_end_relative_s'] <= t_seg['end'])
            seg_preds = h_df[mask]
            correct_preds = seg_preds[seg_preds['predicted_class'] == t_seg['class']]
            if not correct_preds.empty:
                first_pred_time = correct_preds.iloc[0]['window_end_relative_s']
                latency = first_pred_time - t_seg['start']
                latencies.append(latency)
        avg_latency = sum(latencies) / len(latencies) if latencies else float('nan')
        
        results.append({
            "horizon_s": h,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "f1@10": round(f1_10, 4),
            "f1@25": round(f1_25, 4),
            "f1@50": round(f1_50, 4),
            "levenshtein_distance": lev_dist,
            "detection_latency_s": round(avg_latency, 4) if not pd.isna(avg_latency) else None
        })
        
    return pd.DataFrame(results)
