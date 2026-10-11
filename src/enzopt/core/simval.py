"""Simulation + CRNT validation with one automatic re-run (SC5-5).

A run that fails the CRNT checks (conservation drift, negative concentration, Lyapunov increase, equilibrium
relation) is marked "수치 오류 의심" and solved once more with the step reduced to 1/10: for scipy the tolerances and
the maximum step are divided by 10, for fixed-step schemes (Hopsan, explicit Euler) Δt is divided by 10.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .massaction import MassAction, SimulationError, SolverSettings, atol_vector, simulate
from .network import Network
from .validate import validate_closed

SUSPECT = "수치 오류 의심"


@dataclass
class ValidatedRun:
    name: str
    t: np.ndarray
    X: np.ndarray
    report: dict
    retried: bool = False
    first_report: dict | None = None
    step: float | None = None
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.report.get("ok"))

    def row(self) -> dict:
        r = self.report
        return {
            "run": self.name,
            "ok": self.ok,
            "suspect_numerical_error": self.retried,
            "retried_with_step_div10": self.retried,
            "first_ok": None if self.first_report is None else bool(self.first_report.get("ok")),
            "conservation_max_rel_drift": r.get("conservation_max_rel_drift"),
            "min_concentration": r.get("min_concentration"),
            "equilibrium_max_log_error": r.get("equilibrium_max_log_error"),
            "lyapunov_max_increase": r.get("lyapunov_max_increase"),
            "conservation_ok": r.get("conservation_ok"),
            "positivity_ok": r.get("positivity_ok"),
            "equilibrium_ok": r.get("equilibrium_ok"),
            "lyapunov_ok": r.get("lyapunov_ok"),
            "step": self.step,
            "guarantee_checks": r.get("guarantee_checks"),  # False: conservation + non-negativity only (no 149)
        }


def run_validated(
    name: str,
    solve: Callable[[float], tuple[np.ndarray, np.ndarray]],
    species: list[str],
    rxns,
    rates,
    step: float,
    guarantee: bool = True,
) -> ValidatedRun:
    """``solve(step)`` → (t, X). Validate; on failure re-run once with step/10 and keep the second result.
    ``guarantee``: the network has the 149 guarantee (``validate_closed``)."""

    def check(t, X):
        S = {s: X[:, i] for i, s in enumerate(species)}
        return validate_closed(t, S, rxns, rates, guarantee=guarantee)

    try:
        t, X = solve(step)
        rep = check(t, X)
    except SimulationError as e:
        t, X, rep = np.zeros(1), np.zeros((1, len(species))), {"ok": False, "error": str(e)}
    if rep.get("ok"):
        return ValidatedRun(name, t, X, rep, step=step)
    first = rep
    try:
        t2, X2 = solve(step / 10.0)
        rep2 = check(t2, X2)
    except SimulationError as e:
        t2, X2, rep2 = t, X, {"ok": False, "error": str(e)}
    return ValidatedRun(name, t2, X2, rep2, retried=True, first_report=first, step=step / 10.0, notes=[SUSPECT])


def scipy_solver(
    network: Network, k, x0, t_eval, Et: float, base: SolverSettings | None = None, model: MassAction | None = None
) -> Callable[[float], tuple[np.ndarray, np.ndarray]]:
    """solve(step) for scipy: step = 1 → base settings, step = 0.1 → tolerances and max step / 10."""
    base = base or SolverSettings()
    model = model or MassAction.from_network(network)
    t_eval = np.asarray(t_eval, float)

    def solve(step: float):
        st = base
        if step < 1.0:
            st = SolverSettings(
                base.method,
                base.rtol * step,
                base.atol_enzyme_rel * step,
                base.atol_other * step,
                (t_eval[-1] / 100.0) * step,
            )
        X = simulate(
            model, k, x0, t_eval, atol=atol_vector(network.species, network.enzyme_species, Et, st), settings=st
        )
        return t_eval, X

    return solve


def simulate_explicit(model: MassAction, k, x0, t_end: float, dt: float, n_samples: int = 2048):
    """Fixed-step explicit Euler sampled at ``n_samples`` points (a Python stand-in for the Hopsan scheme)."""
    n_steps = int(np.ceil(t_end / dt))
    keep = np.unique(np.linspace(0, n_steps, n_samples).round().astype(int))
    x = np.array(x0, float)
    out, ts = [], []
    j = 0
    for i in range(n_steps + 1):
        if j < len(keep) and i == keep[j]:
            out.append(x.copy())
            ts.append(i * dt)
            j += 1
        if i < n_steps:
            x = x + dt * model.rhs(i * dt, x, k)
    return np.array(ts), np.array(out)


def explicit_solver(model: MassAction, k, x0, t_end: float, n_samples: int = 2048):
    def solve(dt: float):
        return simulate_explicit(model, k, x0, t_end, dt, n_samples)

    return solve


def fastest_rate(model: MassAction, k, x0) -> float:
    """Largest |eigenvalue| of the Jacobian at x0 (sets the Δt of explicit schemes)."""
    J = model.jac(0.0, np.asarray(x0, float), np.asarray(k, float))
    ev = np.linalg.eigvals(J)
    return float(np.max(np.abs(ev))) if len(ev) else 0.0
