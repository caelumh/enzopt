"""Hopsan system-level runs: the apparent one-step class batch (149) and a full-mechanism representative run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.regression import batch149_cases, one_step_network, summarize_batch
from .cli import run_model
from .hmf import network_hmf, write_params


def class_batch(
    out_dir: Path,
    cases: list[tuple[str, dict]],
    k1: float,
    Kp: float,
    t_end: float = 600.0,
    timestep: float = 0.01,
    nsamples: int = 2048,
) -> tuple[dict, Path]:
    """Run the apparent one-step model Pyr + NADH ⇌ Lac + NAD for every initial state via --parameterImport."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = one_step_network()
    k = net.rate_vector([k1], [k1 / Kp])
    hmf = out_dir / "ldh_one_step.hmf"
    hmf.write_text(network_hmf("LDH_one_step", net, k, cases[0][1], t_end, timestep, nsamples), encoding="utf-8")
    results = {}
    for name, x0 in cases:
        pf = out_dir / f"{name}.params.csv"
        write_params(pf, x0)
        t, s = run_model(hmf, out_dir / f"{name}.csv", pf)
        results[name] = (t, s)
    return results, hmf


def batch149_hopsan(out_dir: Path, k1: float = 2.0, Kp: float = 23.0) -> dict:
    """CADKIMM hopsan/batch_class.py (1) + (1b) through HopsanCLI, validated with the CRNT checks."""
    net = one_step_network()
    results, hmf = class_batch(out_dir, batch149_cases(), k1, Kp)
    out = summarize_batch(results, net.reactions(), net.rate_vector([k1], [k1 / Kp]))
    out["backend"] = "hopsan"
    out["hmf"] = str(hmf)
    return out


def full_mechanism_run(
    out_dir: Path, name: str, network, k, x0: dict, t_end: float, timestep: float, nsamples: int = 2048
) -> tuple[np.ndarray, dict, Path]:
    """One full-mechanism run in Hopsan (small Δt, explicit unit-delay scheme)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hmf = out_dir / f"{name}.hmf"
    hmf.write_text(network_hmf(name, network, k, x0, t_end, timestep, nsamples), encoding="utf-8")
    t, s = run_model(hmf, out_dir / f"{name}.csv")
    return t, s, hmf
