import numpy as np
from scipy import signal

from .psd import compute_psd_correlation


def refine_peak_lag_parabolic(
    xcorr: np.ndarray,
    best_idx: int,
    dt: float,
) -> float:
    """Refines the peak lag estimation using 3-point parabolic interpolation.

    Returns the interpolated offset adjustment in milliseconds.
    """
    if 0 < best_idx < len(xcorr) - 1:
        y0 = xcorr[best_idx - 1]
        y1 = xcorr[best_idx]
        y2 = xcorr[best_idx + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-9:
            delta = (y0 - y2) / denom
            return delta * dt * 1000.0
    return 0.0


def compute_window_missingness(
    timestamps_in_window: np.ndarray,
    w_start: float,
    w_end: float,
    nominal_dt: float | None = None,
) -> float:
    """Computes the normalized missingness rate in [0.0, 1.0] for a series within a window.

    Assesses both temporal gap durations exceeding 1.5 * nominal_dt (including boundaries)
    and sample-count deficiency relative to expected nominal count.
    """
    T = w_end - w_start
    if T <= 0:
        return 0.0

    if len(timestamps_in_window) == 0:
        return 1.0

    if nominal_dt is None or nominal_dt <= 0:
        diffs = np.diff(timestamps_in_window)
        diffs = diffs[diffs > 0]
        nominal_dt = float(np.median(diffs)) if len(diffs) > 0 else 0.02

    # 1. Start boundary gap
    missing_time = 0.0
    start_gap = timestamps_in_window[0] - w_start
    if start_gap > 1.5 * nominal_dt:
        missing_time += (start_gap - nominal_dt)

    # 2. Internal gaps between consecutive samples
    if len(timestamps_in_window) > 1:
        gaps = np.diff(timestamps_in_window)
        for g in gaps:
            if g > 1.5 * nominal_dt:
                missing_time += (g - nominal_dt)

    # 3. End boundary gap
    end_gap = w_end - timestamps_in_window[-1]
    if end_gap > 1.5 * nominal_dt:
        missing_time += (end_gap - nominal_dt)

    time_missingness = missing_time / T
    n_expected = max(1.0, T / nominal_dt)
    count_missingness = max(0.0, 1.0 - (len(timestamps_in_window) / n_expected))

    return float(np.clip(max(time_missingness, count_missingness), 0.0, 1.0))


def extract_single_window_data(
    motor_timestamps: np.ndarray,
    motor_data: np.ndarray,
    imu_timestamps: np.ndarray,
    imu_data: np.ndarray,
    start_trial_toa: float,
    window_idx: int,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    resample_rate: float = 200.0,
    max_missingness: float = 0.20,
    min_psd_corr: float = 0.80,
    min_corr: float = 0.5,
    min_std: float = 0.5,
    max_lag_sec: float = 0.5,
) -> dict:
    """Extracts raw signals, assesses missingness and PSD correlation before cross-correlation,
    and computes the cross-correlation curve for a single window.
    """
    w_start_toa = start_trial_toa + window_idx * step_sec
    w_end_toa = w_start_toa + window_sec
    w_start_rel = w_start_toa - start_trial_toa
    w_end_rel = w_end_toa - start_trial_toa

    # Slice raw data within the window
    m_mask = (motor_timestamps >= w_start_toa) & (motor_timestamps <= w_end_toa)
    i_mask = (imu_timestamps >= w_start_toa) & (imu_timestamps <= w_end_toa)

    raw_m_t = motor_timestamps[m_mask] - w_start_toa
    raw_m_val = motor_data[m_mask]

    imu_euler = imu_data[:, 0] if imu_data.ndim > 1 else imu_data
    raw_i_t = imu_timestamps[i_mask] - w_start_toa
    raw_i_val = imu_euler[i_mask]

    raw_m_t_exp = motor_timestamps[m_mask] - start_trial_toa
    raw_i_t_exp = imu_timestamps[i_mask] - start_trial_toa

    # 1. Missingness assessment BEFORE resampling
    # if len(motor_timestamps) > 1:
    #     m_diffs = np.diff(motor_timestamps)
    #     m_nom_dt = float(np.median(m_diffs[m_diffs > 0])) if np.any(m_diffs > 0) else 0.02
    # else:
    #     m_nom_dt = 0.02

    # if len(imu_timestamps) > 1:
    #     i_diffs = np.diff(imu_timestamps)
    #     i_nom_dt = float(np.median(i_diffs[i_diffs > 0])) if np.any(i_diffs > 0) else 0.025
    # else:
    #     i_nom_dt = 0.025

    # motor_missingness = compute_window_missingness(motor_timestamps[m_mask], w_start_toa, w_end_toa, m_nom_dt)
    # imu_missingness = compute_window_missingness(imu_timestamps[i_mask], w_start_toa, w_end_toa, i_nom_dt)
    # window_max_missingness = max(motor_missingness, imu_missingness)

    # 2. Interpolate onto a uniform grid for cross-correlation
    dt = 1.0 / resample_rate
    t_grid = np.arange(w_start_toa, w_end_toa, dt)
    sig_m = np.interp(t_grid, motor_timestamps[m_mask], raw_m_val)
    sig_i = np.interp(t_grid, imu_timestamps[i_mask], raw_i_val)

    m_dm = sig_m - np.mean(sig_m)
    i_dm = sig_i - np.mean(sig_i)
    std_m = np.std(m_dm)
    std_i = np.std(i_dm)

    win_samples = len(t_grid)
    max_lag_samples = int(round(max_lag_sec * resample_rate))

    # 3. Compute PSD correlation assessment
    psd_corr = compute_psd_correlation(sig_m, sig_i, fs=resample_rate, min_std=min_std)
    is_active = (std_m >= min_std) and (std_i >= min_std)

    if std_m > 1e-6 and std_i > 1e-6:
        xcorr = signal.correlate(m_dm, i_dm, mode="full") / (win_samples * std_m * std_i)
        lags_samples = signal.correlation_lags(win_samples, win_samples, mode="full")
        lags_ms = lags_samples * dt * 1000.0

        lag_mask = np.abs(lags_samples) <= max_lag_samples
        restricted_xcorr = xcorr[lag_mask]
        restricted_lags = lags_ms[lag_mask]

        best_idx = np.argmax(restricted_xcorr)
        peak_corr = restricted_xcorr[best_idx]
        best_lag_ms = restricted_lags[best_idx]

        # Parabolic interpolation around the peak for sub-sample accuracy
        best_lag_ms += refine_peak_lag_parabolic(restricted_xcorr, best_idx, dt)
    else:
        restricted_xcorr = np.zeros(2 * max_lag_samples + 1)
        restricted_lags = np.linspace(-max_lag_sec * 1000, max_lag_sec * 1000, len(restricted_xcorr))
        peak_corr = 0.0
        best_lag_ms = 0.0

    # 4. Inclusion decision
    # if window_max_missingness > max_missingness:
    #     is_included = False
    #     exclusion_reason = "HIGH_MISSINGNESS"
    if not is_active:
        is_included = False
        exclusion_reason = "STATIONARY"
    elif psd_corr < min_psd_corr:
        is_included = False
        exclusion_reason = "LOW_PSD_CORRELATION"
    elif peak_corr < min_corr:
        is_included = False
        exclusion_reason = "LOW_TIME_CORRELATION"
    else:
        is_included = True
        exclusion_reason = "PASSED"

    return {
        "window_idx": window_idx,
        "w_start_rel": w_start_rel,
        "w_end_rel": w_end_rel,
        "raw_m_t": raw_m_t,
        "raw_m_t_exp": raw_m_t_exp,
        "raw_m_val": raw_m_val,
        "raw_i_t": raw_i_t,
        "raw_i_t_exp": raw_i_t_exp,
        "raw_i_val": raw_i_val,
        "lags_ms": restricted_lags,
        "xcorr": restricted_xcorr,
        "peak_lag_ms": best_lag_ms,
        "peak_corr": peak_corr,
        "psd_corr": psd_corr,
        # "motor_missingness": motor_missingness,
        # "imu_missingness": imu_missingness,
        # "max_missingness": window_max_missingness,
        "offset_ms": abs(best_lag_ms),
        "is_included": is_included,
        "exclusion_reason": exclusion_reason,
    }


def run_cross_correlation_analysis(
    motor_timestamps: np.ndarray,
    motor_data: np.ndarray,
    imu_timestamps: np.ndarray,
    imu_data: np.ndarray,
    start_trial_toa: float,
    end_trial_toa: float,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    resample_rate: float = 200.0,
    max_missingness: float = 0.20,
    min_psd_corr: float = 0.80,
    min_corr: float = 0.5,
    min_std: float = 0.5,
    include_all: bool = False,
    max_lag_sec: float = 0.5,
) -> dict:
    """Slices motor and IMU data into overlapping windows, assesses missingness and spectral
    similarity via PSD before cross-correlation, and performs cross-correlation to find relative offset.
    """
    duration = end_trial_toa - start_trial_toa
    dt = 1.0 / resample_rate
    win_samples = int(round(window_sec * resample_rate))
    max_lag_samples = int(round(max_lag_sec * resample_rate))

    # Pre-calculate nominal sampling periods for missingness evaluation
    diffs_m = np.diff(motor_timestamps)
    diffs_m = diffs_m[diffs_m > 0]
    motor_nominal_dt = float(np.median(diffs_m)) if len(diffs_m) > 0 else 0.02

    diffs_i = np.diff(imu_timestamps)
    diffs_i = diffs_i[diffs_i > 0]
    imu_nominal_dt = float(np.median(diffs_i)) if len(diffs_i) > 0 else 0.025

    # Pre-resample the entire trial to a regular grid for fast window processing
    t_grid_full = np.arange(start_trial_toa, end_trial_toa, dt)
    motor_interp = np.interp(t_grid_full, motor_timestamps, motor_data)
    imu_euler = imu_data[:, 0] if imu_data.ndim > 1 else imu_data
    imu_interp = np.interp(t_grid_full, imu_timestamps, imu_euler)

    step_samples = int(round(step_sec * resample_rate))
    num_windows = (len(t_grid_full) - win_samples) // step_samples + 1

    window_indices = []
    window_start_times = []
    window_end_times = []
    window_start_toas = []
    window_end_toas = []
    offsets_ms = []        # Magnitude of offset in milliseconds
    signed_lags_ms = []    # Signed lag in milliseconds
    correlations = []
    psd_correlations = []
    # motor_missingness_list = []
    # imu_missingness_list = []
    # max_missingness_list = []
    std_motors = []
    std_imus = []
    is_included_list = []
    exclusion_reasons = []

    for i in range(num_windows):
        idx_start = i * step_samples
        idx_end = idx_start + win_samples
        if idx_end > len(t_grid_full):
            break

        w_start_toa = t_grid_full[idx_start]
        w_end_toa = t_grid_full[idx_end - 1]
        w_start_rel = w_start_toa - start_trial_toa
        w_end_rel = w_end_toa - start_trial_toa

        # Missingness assessment before resampling / cross-correlation
        # m_win_mask = (motor_timestamps >= w_start_toa) & (motor_timestamps <= w_end_toa)
        # i_win_mask = (imu_timestamps >= w_start_toa) & (imu_timestamps <= w_end_toa)

        # m_miss = compute_window_missingness(motor_timestamps[m_win_mask], w_start_toa, w_end_toa, motor_nominal_dt)
        # i_miss = compute_window_missingness(imu_timestamps[i_win_mask], w_start_toa, w_end_toa, imu_nominal_dt)
        # w_miss = max(m_miss, i_miss)

        sig_m = motor_interp[idx_start:idx_end]
        sig_i = imu_interp[idx_start:idx_end]

        m_dm = sig_m - np.mean(sig_m)
        i_dm = sig_i - np.mean(sig_i)

        std_m = np.std(m_dm)
        std_i = np.std(i_dm)

        # Check for movement/activity
        is_active = (std_m >= min_std) and (std_i >= min_std)

        if std_m < 1e-6 or std_i < 1e-6:
            window_indices.append(i)
            window_start_times.append(w_start_rel)
            window_end_times.append(w_end_rel)
            window_start_toas.append(w_start_toa)
            window_end_toas.append(w_end_toa)
            offsets_ms.append(0.0)
            signed_lags_ms.append(0.0)
            correlations.append(0.0)
            psd_correlations.append(0.0)
            # motor_missingness_list.append(m_miss)
            # imu_missingness_list.append(i_miss)
            # max_missingness_list.append(w_miss)
            std_motors.append(float(std_m))
            std_imus.append(float(std_i))
            is_included_list.append(False)
            # if w_miss > max_missingness:
            #     exclusion_reasons.append("HIGH_MISSINGNESS")
            # else:
            exclusion_reasons.append("STATIONARY")
            continue

        # PSD correlation assessment (lag-invariant similarity)
        psd_corr = compute_psd_correlation(sig_m, sig_i, fs=resample_rate, min_std=min_std)

        # Normalized cross-correlation
        xcorr = signal.correlate(m_dm, i_dm, mode="full") / (win_samples * std_m * std_i)
        lags = signal.correlation_lags(win_samples, win_samples, mode="full")

        # Restrict search within allowable max_lag
        lag_mask = np.abs(lags) <= max_lag_samples
        restricted_xcorr = xcorr[lag_mask]
        restricted_lags = lags[lag_mask]

        best_sub_idx = np.argmax(restricted_xcorr)
        peak_corr = restricted_xcorr[best_sub_idx]
        best_lag_samples = restricted_lags[best_sub_idx]

        # Parabolic peak interpolation for sub-sample precision
        refined_lag_samples = float(best_lag_samples)
        if 0 < best_sub_idx < len(restricted_xcorr) - 1:
            y0 = restricted_xcorr[best_sub_idx - 1]
            y1 = restricted_xcorr[best_sub_idx]
            y2 = restricted_xcorr[best_sub_idx + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            if abs(denom) > 1e-9:
                delta = (y0 - y2) / denom
                refined_lag_samples += delta

        best_lag_s = refined_lag_samples * dt
        best_lag_ms = best_lag_s * 1e3
        offset_magnitude_ms = abs(best_lag_ms)

        # Determine inclusion based on missingness, activity, PSD correlation, and cross-correlation
        # if w_miss > max_missingness:
        #     is_included = False
        #     exclusion_reason = "HIGH_MISSINGNESS"
        if not is_active:
            is_included = False
            exclusion_reason = "STATIONARY"
        elif psd_corr < min_psd_corr:
            is_included = False
            exclusion_reason = "LOW_PSD_CORRELATION"
        elif peak_corr < min_corr:
            is_included = False
            exclusion_reason = "LOW_TIME_CORRELATION"
        else:
            is_included = True
            exclusion_reason = "PASSED"

        window_indices.append(i)
        window_start_times.append(w_start_rel)
        window_end_times.append(w_end_rel)
        window_start_toas.append(w_start_toa)
        window_end_toas.append(w_end_toa)
        offsets_ms.append(offset_magnitude_ms)
        signed_lags_ms.append(best_lag_ms)
        correlations.append(peak_corr)
        psd_correlations.append(psd_corr)
        # motor_missingness_list.append(m_miss)
        # imu_missingness_list.append(i_miss)
        # max_missingness_list.append(w_miss)
        std_motors.append(float(std_m))
        std_imus.append(float(std_i))
        is_included_list.append(is_included)
        exclusion_reasons.append(exclusion_reason)

    return {
        "window_indices": np.array(window_indices),
        "start_times": np.array(window_start_times),
        "end_times": np.array(window_end_times),
        "start_toas": np.array(window_start_toas),
        "end_toas": np.array(window_end_toas),
        "offsets_ms": np.array(offsets_ms),
        "signed_lags_ms": np.array(signed_lags_ms),
        "correlations": np.array(correlations),
        "psd_correlations": np.array(psd_correlations),
        # "motor_missingness": np.array(motor_missingness_list),
        # "imu_missingness": np.array(imu_missingness_list),
        # "max_missingness": np.array(max_missingness_list),
        "std_motors": np.array(std_motors),
        "std_imus": np.array(std_imus),
        "is_included": np.array(is_included_list),
        "is_valid": np.array(is_included_list),
        "exclusion_reasons": np.array(exclusion_reasons),
        "duration": duration,
    }
