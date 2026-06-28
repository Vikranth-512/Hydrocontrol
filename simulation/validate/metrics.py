"""
RL benchmark difficulty metrics: observability, controllability, delay estimation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def mutual_information_binned(
    x: np.ndarray, y: np.ndarray, bins: int = 20
) -> float:
    """
    Estimate mutual information I(X;Y) via histogram binning.
    Higher MI means the hidden state is more observable from the sensor.
    """
    eps = 1e-12
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    c_xy = c_xy / (c_xy.sum() + eps)
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i, j] > eps:
                mi += c_xy[i, j] * np.log(c_xy[i, j] / (c_x[i] * c_y[j] + eps) + eps)
    return float(mi)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation between two signals for lags in [-max_lag, max_lag].
    Returns (lags, correlation_values).
    """
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            corr[i] = np.mean(x[:n - lag] * y[lag:])
        else:
            corr[i] = np.mean(x[-lag:] * y[:n + lag])
    return lags, corr


def estimate_impulse_delay(
    signal: np.ndarray, impulse_time_idx: int, threshold_frac: float = 0.1
) -> int:
    """
    Estimate the delay (in timesteps) from an impulse injection point
    to when a signal first rises above threshold_frac * its eventual peak.
    """
    post = signal[impulse_time_idx:]
    baseline = post[0]
    peak = np.max(post) - baseline
    if peak < 1e-12:
        return -1  # No detectable response
    threshold = baseline + threshold_frac * peak
    crossings = np.where(post > threshold)[0]
    if len(crossings) == 0:
        return -1
    return int(crossings[0])
