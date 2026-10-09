"""Robust assay design (CRNT equilibrium limit + forward simulation over rate-constant samples).

Ported from CADKIMM ``analysis/design_optimize.py`` and generalised to the project's design network.

Problem: quantify an analyte (NADH, ``analyte0`` mM) by consuming ``conversion`` of it within ``time_s`` with the
enzyme, choosing the partner ratio ρ = partner₀/analyte₀ and the smallest enzyme amount Et that works for every
rate-constant sample (robust) — or for the nominal sample only (nominal, to show why robustness matters).

CRNT gives the equilibrium limit: weakly reversible + deficiency 0 → one equilibrium per compatibility class, fixed
by K' whatever the rate constants. With no products at t = 0 the equilibrium conversion f solves
K' = f² / ((ρ − f)(1 − f)); the smallest ρ that allows ``conversion`` is ρ_min = c + c²/((1 − c)K').
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .massaction import MassAction, SimulationError, SolverSettings, haldane_kr, simulate
from .network import Network, x0_vector

# CADKIMM placeholder constants (design_optimize.py): used for the synthetic-sample regression only
KF0 = np.array([1e4, 5e3, 300.0, 1000.0, 200.0, 1e3])
KR0 = np.array([50.0, 500.0, 1000.0, 200.0, np.nan, 1e3 * 5.0])  # k-6 = k+6 · Ki, Ki = 5 mM (assumed)


@dataclass
class AssayDesign:
    network: Network
    analyte: str = "NADH"
    partner: str = "Pyr"
    free_enzyme: str = "E"
    analyte0: float = 0.1
    conversion: float = 0.99
    time_s: float = 300.0
    Kp: float = 23.0
    rtol: float = 1e-7
    atol_enzyme_rel: float = 1e-9
    atol_other: float = 1e-12
    _model: MassAction | None = field(default=None, repr=False, compare=False)

    @property
    def model(self) -> MassAction:
        if self._model is None:
            self._model = MassAction.from_network(self.network)
        return self._model

    @property
    def target(self) -> float:
        return (1.0 - self.conversion) * self.analyte0

    def x0(self, Et: float, rho: float) -> np.ndarray:
        return x0_vector(
            self.network, {self.analyte: self.analyte0, self.partner: rho * self.analyte0}, Et, self.free_enzyme
        )

    def atol(self, Et: float) -> np.ndarray:
        ens = set(self.network.enzyme_species)
        return np.array([Et * self.atol_enzyme_rel if s in ens else self.atol_other for s in self.network.species])

    def trajectory(self, k, Et: float, rho: float, t_eval) -> np.ndarray:
        st = SolverSettings(method="Radau", rtol=self.rtol)
        return simulate(self.model, k, self.x0(Et, rho), np.asarray(t_eval, float), atol=self.atol(Et), settings=st)

    def analyte_at(self, k, Et: float, rho: float, t_end: float | None = None) -> float:
        t_end = self.time_s if t_end is None else t_end
        try:
            X = self.trajectory(k, Et, rho, [t_end])
        except SimulationError:
            return float("nan")
        return float(X[-1, self.network.index()[self.analyte]])


def rho_min_equilibrium(conv: float, Kp: float) -> float:
    return conv + conv**2 / ((1 - conv) * Kp)


def equilibrium_conversion(rho: float, Kp: float) -> float:
    """Root 0 < f < min(1, ρ) of K' = f²/((ρ − f)(1 − f)) (no products at t = 0). Independent of rate constants
    and enzyme amount (CRNT, weakly reversible + deficiency 0): no amount of enzyme can go beyond it."""
    a, b, c = Kp - 1.0, -Kp * (1.0 + rho), Kp * rho
    if abs(a) < 1e-12:
        return -c / b
    disc = np.sqrt(b * b - 4 * a * c)
    roots = [(-b - disc) / (2 * a), (-b + disc) / (2 * a)]
    return float([r for r in roots if 0 < r < min(1.0, rho) + 1e-12][0])


def param_samples(n: int, seed: int = 3, spread: float = 0.5, Kp: float = 23.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """CADKIMM placeholder samples: each constant shaken within ±spread decades, k-5 recomputed by Haldane."""
    rng = np.random.default_rng(seed)
    out = [(KF0.copy(), haldane_kr(KF0, KR0, Kp, 5))]
    for _ in range(n - 1):
        kf = KF0 * 10 ** rng.uniform(-spread, spread, len(KF0))
        kr = KR0 * 10 ** rng.uniform(-spread, spread, len(KR0))
        out.append((kf, haldane_kr(kf, kr, Kp, 5)))
    return out


def min_Et(
    design: AssayDesign, rho: float, samples: list[np.ndarray], lo: float = 1e-6, hi: float = 1e-2, iters: int = 18
) -> float:
    """Smallest Et with analyte(time_s) ≤ (1 − conversion)·analyte0 for every sample (log bisection)."""
    target = design.target

    def ok(Et):
        return all(design.analyte_at(k, Et, rho) <= target for k in samples)

    if not ok(hi):
        return float("nan")
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def _row(args):
    design, rho, samples, lo, hi, iters = args
    return rho, min_Et(design, rho, samples[:1], lo, hi, iters), min_Et(design, rho, samples, lo, hi, iters)


def default_rho_grid(rho_min: float, quick: bool) -> list[float]:
    if quick:
        return [rho_min * 1.05, 8.0, 15.0, 30.0]
    return [rho_min * 1.02, rho_min * 1.1, 7.0, 9.0, 12.0, 16.0, 22.0, 30.0, 45.0, 70.0, 100.0]


def n_workers(n_tasks: int) -> int:
    env = os.environ.get("ENZOPT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(n_tasks, (os.cpu_count() or 2) - 2))


@dataclass
class DesignResult:
    rho_min: float | None
    rhos: list[float]
    Et_nominal: list[float]
    Et_robust: list[float]
    rho_star: float
    Et_star: float
    Et_nominal_at_star: float
    check: list[dict]
    equilibrium_limit: list[dict]

    def summary(self) -> dict:
        nom = [c for c in self.check if c["design"] == "nominal"]
        rob = [c for c in self.check if c["design"] == "robust"]
        return {
            "rho_min": self.rho_min,
            "rho_star": self.rho_star,
            "robust_Et_star_mM": self.Et_star,
            "robust_Et_star_nM": self.Et_star * 1e6,
            "nominal_Et_at_rho_star_mM": self.Et_nominal_at_star,
            "nominal_Et_at_rho_star_nM": self.Et_nominal_at_star * 1e6,
            "nominal_fail": sum(c["fail"] for c in nom),
            "nominal_worst_conversion": min(c["conversion"] for c in nom) if nom else None,
            "robust_fail": sum(c["fail"] for c in rob),
            "robust_worst_conversion": min(c["conversion"] for c in rob) if rob else None,
            "n_samples": len(rob),
        }


def run_design(
    design: AssayDesign,
    samples: list[np.ndarray],
    rhos: list[float],
    iters: int = 18,
    Et_bounds: tuple[float, float] = (1e-6, 1e-2),
    use_equilibrium_limit: bool = True,
    workers: int | None = None,
    progress=None,
) -> DesignResult:
    """U-curve over ``rhos`` (nominal = first sample, robust = all samples), optimum, and a check of both designs
    on every sample. ``samples`` are rate vectors aligned with ``design.network.reactions()``."""
    lo, hi = Et_bounds
    rmin = rho_min_equilibrium(design.conversion, design.Kp) if use_equilibrium_limit else None
    eq_rows = []
    if use_equilibrium_limit:
        for r in [1.0, 1.5, 2.0, 3.0, rmin, 8.0]:
            f = equilibrium_conversion(r, design.Kp)
            eq_rows.append({"rho": r, "conversion_eq": f, "NADH_left_uM": (1 - f) * design.analyte0 * 1e3})
    tasks = [(design, float(r), samples, lo, hi, iters) for r in rhos]
    w = workers if workers is not None else n_workers(len(tasks))
    rows = []
    if w > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=w) as ex:
            for i, r in enumerate(ex.map(_row, tasks)):
                rows.append(r)
                if progress:
                    progress((i + 1) / (len(tasks) + 1), f"ρ {i + 1}/{len(tasks)}")
    else:
        for i, t in enumerate(tasks):
            rows.append(_row(t))
            if progress:
                progress((i + 1) / (len(tasks) + 1), f"ρ {i + 1}/{len(tasks)}")
    arr = np.array(rows, float)
    if np.all(np.isnan(arr[:, 2])):
        raise SimulationError("모든 ρ에서 효소량 상한 안에 목표를 만족하는 설계가 없음")
    j = int(np.nanargmin(arr[:, 2]))
    rho_s, Et_nom, Et_rob = arr[j]
    check = []
    for label, Et in (("nominal", Et_nom), ("robust", Et_rob)):
        for i, k in enumerate(samples):
            x = design.analyte_at(k, Et, rho_s) if np.isfinite(Et) else float("nan")
            check.append(
                {
                    "design": label,
                    "sample": i,
                    "Et_mM": float(Et),
                    "NADH_at_target_mM": x,
                    "conversion": 1 - x / design.analyte0,
                    "fail": bool(not np.isfinite(x) or x > design.target * (1 + 1e-6)),
                }
            )
    return DesignResult(
        rmin,
        [float(r) for r in arr[:, 0]],
        [float(v) for v in arr[:, 1]],
        [float(v) for v in arr[:, 2]],
        float(rho_s),
        float(Et_rob),
        float(Et_nom),
        check,
        eq_rows,
    )
