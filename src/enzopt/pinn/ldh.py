"""LDH ordered bi-bi defaults for the general PINN (``pinn.general``).

Until F-13 this module held a PINN written for the LDH mechanism only (9 species, 5 reversible steps, King–Altman ring).
The general PINN covers it (decision 2026-10-10: one code path); this module keeps the LDH names and fills the LDH
network into a ``PinnConfig`` without one, so older callers (scripts, tests) keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from ..core.activity import activity_curve, activity_rate  # noqa: F401 (re-exported)
from ..core.network import Network
from .general import (  # noqa: F401 (re-exported)
    GeneralPinn,
    PinnConfig,
    PinnExperiment,
    PinnResult,
    cuda_usable,
    resolve_device,
    supports,
)
from .general import train_pinn as _train_general

SPECIES = ["E", "NADH", "EN", "Pyr", "ENP", "EQL", "Lac", "EQ", "NAD"]
STEPS = [
    ({"E": 1, "NADH": 1}, {"EN": 1}),
    ({"EN": 1, "Pyr": 1}, {"ENP": 1}),
    ({"ENP": 1}, {"EQL": 1}),
    ({"EQL": 1}, {"EQ": 1, "Lac": 1}),
    ({"EQ": 1}, {"E": 1, "NAD": 1}),
]
STEP_TEXT = ["E + NADH <=> EN", "EN + Pyr <=> ENP", "ENP <=> EQL", "EQL <=> EQ + Lac", "EQ <=> E + NAD"]
ENZYME = ["E", "EN", "ENP", "EQL", "EQ"]
OVERALL = "NADH + Pyr <=> Lac + NAD"
N_THETA_WR = 10  # log10 kf(5), log10 kr1..kr4, log10 K'
N_THETA_IRR = 9  # last step one-way: no K'


def matches_network(network) -> bool:
    """True if ``network`` is the LDH ordered bi-bi mechanism (step order as in STEPS)."""
    if sorted(network.species) != sorted(SPECIES) or len(network.steps) != len(STEPS):
        return False
    for st, (a, b) in zip(network.steps, STEPS, strict=True):
        if st.lhs != a or st.rhs != b:
            return False
    return all(st.reversible for st in network.steps[:4])


def ldh_network(irreversible_last: bool = False) -> Network:
    steps = list(STEP_TEXT)
    if irreversible_last:
        steps[-1] = steps[-1].replace("<=>", "->")
    return Network.from_strings("ldh_ordered_bibi", SPECIES, steps, ENZYME)


def with_ldh_defaults(cfg: PinnConfig) -> PinnConfig:
    """``cfg`` with the LDH network, A340 observable and overall reaction filled in where missing."""
    if cfg.network is not None:
        return cfg
    eps = cfg.eps340
    return replace(
        cfg,
        network=ldh_network(cfg.irreversible_last),
        observable=cfg.observable or {"NADH": eps, "EN": eps, "ENP": eps},
        overall=cfg.overall or [OVERALL],
        prefer_dependent=cfg.prefer_dependent or [5],
    )


def train_pinn(
    exps: list[PinnExperiment],
    prior_center: np.ndarray,
    cfg: PinnConfig,
    progress: Callable[[float, str], None] | None = None,
) -> list[PinnResult]:
    return _train_general(exps, prior_center, with_ldh_defaults(cfg), progress)
