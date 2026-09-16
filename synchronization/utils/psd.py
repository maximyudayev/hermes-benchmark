import numpy as np
from scipy import signal


def compute_psd_correlation(
    sig_m: np.ndarray,
    sig_i: np.ndarray,
    fs: float,
    min_std: float = 1e-6,
) -> float:
    """Computes the Pearson correlation coefficient between the Power Spectral Densities (PSDs)
    of two demeaned signals using Welch's method.

    This provides a lag-invariant assessment of spectral similarity to detect
    artifacts, sensor dropouts, or disconnections before cross-correlation.
    """
    m_dm = sig_m - np.mean(sig_m)
    i_dm = sig_i - np.mean(sig_i)
    std_m = np.std(m_dm)
    std_i = np.std(i_dm)

    if std_m < min_std or std_i < min_std:
        return 0.0

    nperseg = min(len(sig_m), 256)
    _, psd_m = signal.welch(m_dm, fs=fs, nperseg=nperseg)
    _, psd_i = signal.welch(i_dm, fs=fs, nperseg=nperseg)

    std_psd_m = np.std(psd_m)
    std_psd_i = np.std(psd_i)
    if std_psd_m < 1e-12 or std_psd_i < 1e-12:
        return 0.0

    corr = np.corrcoef(psd_m, psd_i)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)
