"""Criterion-based choice of PINN rate-constant samples (identification.pinn.selection: criteria).

Starting points replace the random jitter: the first n points of the unscrambled Sobol sequence, moved to the middle
of their strata and mapped to prior centre ± half_width decades. The design is deterministic and space-filling: for
n = 2^m every coordinate takes the n evenly spaced stratum midpoints exactly once.

Samples are then chosen among the trained candidates by two criteria:

1. Qualification (equally good fit): the ODE training RMSE lies within a likelihood-ratio threshold of the best
   candidate. With Gaussian errors of unknown common variance and n_eff independent residuals,
   −2 ln Λ = n_eff · ln(RMSE² / RMSE_best²); a candidate qualifies when this stays below the χ²(1) quantile
   (the profile-likelihood threshold for one scalar quantity, e.g. the designed enzyme amount).
2. Spread (maximin): sample 0 is the best fit; each further sample is the qualified candidate farthest from the
   samples already chosen (Euclidean distance of log10 constants in prior standard deviations).

numpy and scipy only (KIMM-CAD compatible: no numpy-2-only API).
"""

from __future__ import annotations

import numpy as np

CHI2_1_95 = 3.841458820694124  # χ²(1) 95 % quantile


def start_offsets(n: int, d: int, half_width: float) -> np.ndarray:
    """(n, d) offsets in decades from the prior centre: midpoint-shifted unscrambled Sobol points in ±half_width."""
    if n < 1 or d < 1:
        raise ValueError(f"start_offsets: n = {n}, d = {d} (both ≥ 1)")
    from scipy.stats import qmc

    m = max(0, int(np.ceil(np.log2(n))))
    pts = qmc.Sobol(d, scramble=False).random_base2(m)[:n]
    u = pts + 0.5 / 2**m
    return half_width * (2.0 * u - 1.0)


def lr_threshold(best_rmse: float, n_eff: int, quantile: float = CHI2_1_95) -> float:
    """Largest RMSE whose likelihood ratio to the best fit stays inside ``quantile``: best · exp(q / (2 n_eff))."""
    if n_eff < 1:
        raise ValueError(f"lr_threshold: n_eff = {n_eff} (≥ 1)")
    return float(best_rmse * np.exp(quantile / (2.0 * n_eff)))


def maximin_select(
    theta: np.ndarray, rmse: np.ndarray, qualified: np.ndarray, k: int, scale: float = 1.0
) -> list[tuple[int, float | None]]:
    """Greedy maximin choice of up to ``k`` qualified candidates.

    The best fit (lowest RMSE) comes first; then, repeatedly, the qualified candidate whose distance to the nearest
    chosen one (Euclidean, ``theta / scale``) is largest. Ties go to the lower RMSE, then to the lower index.
    Returns [(candidate index, distance to the nearest earlier choice, None for the first)]."""
    theta = np.asarray(theta, float)
    rmse = np.asarray(rmse, float)
    pool = [i for i in range(len(rmse)) if bool(qualified[i]) and np.isfinite(rmse[i])]
    if not pool or k < 1:
        return []
    z = theta / float(scale)
    first = min(pool, key=lambda i: (rmse[i], i))
    chosen: list[tuple[int, float | None]] = [(first, None)]
    nearest = {i: float(np.linalg.norm(z[i] - z[first])) for i in pool if i != first}
    while len(chosen) < k and nearest:
        best = max(nearest, key=lambda i: (round(nearest[i], 9), -rmse[i], -i))
        chosen.append((best, nearest.pop(best)))
        for i in nearest:
            nearest[i] = min(nearest[i], float(np.linalg.norm(z[i] - z[best])))
    return chosen
