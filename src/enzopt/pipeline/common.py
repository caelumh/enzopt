"""Helpers shared by the stages: rate vectors from samples, observable curves, sample files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.massaction import MassAction, SimulationError, SolverSettings, atol_vector, simulate
from ..core.network import Network, x0_vector


def observable(network: Network, X: np.ndarray, coefficients: dict) -> np.ndarray:
    idx = network.index()
    return sum(c * X[:, idx[s]] for s, c in coefficients.items())


def curve(
    network: Network,
    k,
    initial: dict,
    Et: float,
    t,
    coefficients: dict,
    model: MassAction | None = None,
    settings: SolverSettings | None = None,
) -> np.ndarray:
    """Observable at times ``t`` (simulation starts at t = 0 from ``initial`` with free enzyme Et)."""
    st = settings or SolverSettings()
    model = model or MassAction.from_network(network)
    t = np.asarray(t, float)
    t_eval = t if t[0] > 0 else t
    X = simulate(
        model,
        k,
        x0_vector(network, initial, Et),
        t_eval,
        atol=atol_vector(network.species, network.enzyme_species, Et, st),
        settings=st,
    )
    return observable(network, X, coefficients)


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def safe_curve(*args, **kw):
    try:
        return curve(*args, **kw)
    except (SimulationError, ValueError, FloatingPointError):
        return None


def load_samples(run_dir: Path, kept_only: bool = True) -> list[dict]:
    d = Path(run_dir) / "3_fit" / "samples"
    out = []
    for f in sorted(d.glob("sample_seed*.json"), key=lambda p: int(p.stem.replace("sample_seed", ""))):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("kept", True) or not kept_only:
            out.append(s)
    return out


def sample_rates(network: Network, sample: dict) -> np.ndarray:
    return network.rate_vector(sample["kf"], sample["kr"])


def design_rates(design_network: Network, sample: dict, design_cfg: dict, Kp: float, haldane_step: int | None):
    """Rate vector for the design network: identified steps + dead-end step; k₋(haldane) recomputed with design K'."""
    from ..core.massaction import haldane_kr

    kf = list(sample["kf"])
    kr = list(sample["kr"])
    if haldane_step:
        kr = list(haldane_kr(np.array(kf, float), np.array(kr, float), Kp, haldane_step))
    de = design_cfg.get("dead_end")
    if de:
        kf.append(float(de["kf"]))
        kr.append(float(de["kf"]) * float(de["Ki_mM"]))
    return design_network.rate_vector(kf, kr)
