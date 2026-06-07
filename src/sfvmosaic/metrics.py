from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def mssim(a: np.ndarray, b: np.ndarray, window: int) -> float:
    """Block-window MSSIM, averaged over independent RGB channel SSIM values."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    values = []
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    for y in range(0, a.shape[0], window):
        for x in range(0, a.shape[1], window):
            aa, bb = a[y : y + window, x : x + window], b[y : y + window, x : x + window]
            for channel in range(aa.shape[2]):
                ac, bc = aa[:, :, channel], bb[:, :, channel]
                mux, muy = ac.mean(), bc.mean()
                vx, vy = ac.var(), bc.var()
                covariance = ((ac - mux) * (bc - muy)).mean()
                values.append(
                    ((2 * mux * muy + c1) * (2 * covariance + c2))
                    / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
                )
    return float(np.mean(values))
