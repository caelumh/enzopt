"""Robust assay design (CRNT equilibrium limit + forward simulation over rate-constant samples).

Ported from CADKIMM ``analysis/design_optimize.py`` and generalised to the project's design network.

Problem: quantify an analyte (NADH, ``analyte0`` mM) by consuming ``conversion`` of it within ``time_s`` with the
enzyme, choosing the partner ratio ρ = partner₀/analyte₀ and the smallest enzyme amount Et that works for every
rate-constant sample (robust) — or for the nominal sample only (nominal, to show why robustness matters).

CRNT gives the equilibrium limit: weakly reversible + deficiency 0 → one equilibrium per compatibility class, fixed
by K' whatever the rate constants. With no products at t = 0 the equilibrium conversion f solves
K' = f² / ((ρ − f)(1 − f)); the smallest ρ that allows ``conversion`` is ρ_min = c + c²/((1 − c)K').

F-13 (G5): any overall reaction (``overall``: the equilibrium of A + B + C ⇌ P + Q + R … is solved numerically, so one,
two or three substrates and any stoichiometry), further partners at fixed multiples of analyte₀ (``fixed``), one
substrate (no partner: only the enzyme amount is chosen), and other goals: ``yield`` (product formed ≥ yield ×
analyte₀), ``selectivity`` (product / (product + side product) ≥ s and the yield), ``time`` (with the enzyme amount
fixed, the ρ that reaches the conversion soonest in the worst sample).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .activity import enzyme_loss
from .massaction import MassAction, SimulationError, SolverSettings, haldane_kr, simulate
from .network import Network, Step, x0_vector

GOALS = ("assay", "yield", "selectivity", "time", "peak")

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
    fixed: dict = field(default_factory=dict)  # further species at t = 0, as multiples of analyte₀
    goal: str = "assay"  # assay | yield | selectivity | time
    product: str | None = None  # yield / selectivity: the wanted product
    side_product: str | None = None  # selectivity: the unwanted product
    yield_min: float = 0.9
    selectivity_min: float = 0.9
    overall: str | None = None  # overall reaction of the analyte (numerical equilibrium limit); None = LDH formula

    @property
    def model(self) -> MassAction:
        if self._model is None:
            self._model = MassAction.from_network(self.network)
        return self._model

    @property
    def target(self) -> float:
        return (1.0 - self.conversion) * self.analyte0

    def initial(self, rho: float) -> dict:
        d = {self.analyte: self.analyte0}
        if self.partner:
            d[self.partner] = rho * self.analyte0
        for sp, m in (self.fixed or {}).items():
            d[sp] = float(m) * self.analyte0
        return d

    def x0(self, Et: float, rho: float) -> np.ndarray:
        return x0_vector(self.network, self.initial(rho), Et, self.free_enzyme)

    @property
    def product_per_analyte(self) -> float:
        """Product formed per analyte consumed in the overall reaction (1 without one)."""
        if not (self.overall and self.product):
            return 1.0
        st = Step.parse(self.overall)
        a = st.lhs.get(self.analyte) or st.rhs.get(self.analyte) or 1.0
        p_ = st.rhs.get(self.product) or st.lhs.get(self.product) or 1.0
        return float(p_) / float(a)

    def atol(self, Et: float) -> np.ndarray:
        ens = set(self.network.enzyme_species)
        return np.array([Et * self.atol_enzyme_rel if s in ens else self.atol_other for s in self.network.species])

    def trajectory(self, k, Et: float, rho: float, t_eval, activity: dict | None = None) -> np.ndarray:
        st = SolverSettings(method="Radau", rtol=self.rtol)
        loss = enzyme_loss(self.network.species, self.network.enzyme_species, activity)
        x0 = self.x0(Et, rho)
        return simulate(self.model, k, x0, np.asarray(t_eval, float), atol=self.atol(Et), settings=st, loss=loss)

    def analyte_at(self, k, Et: float, rho: float, t_end: float | None = None, activity: dict | None = None) -> float:
        t_end = self.time_s if t_end is None else t_end
        try:
            X = self.trajectory(k, Et, rho, [t_end], activity)
        except SimulationError:
            return float("nan")
        return float(X[-1, self.network.index()[self.analyte]])

    def evaluate(self, k, Et: float, rho: float, activity: dict | None = None) -> dict:
        """Analyte left, conversion, yield and selectivity at ``time_s`` and whether the goal is met."""
        try:
            X = self.trajectory(k, Et, rho, [self.time_s], activity)[-1]
        except SimulationError:
            return {"analyte": float("nan"), "conversion": float("nan"), "ok": False}
        i = self.network.index()
        x = float(X[i[self.analyte]])
        out = {"analyte": x, "conversion": 1 - x / self.analyte0}
        if self.product:
            P = float(X[i[self.product]])
            out["yield"] = P / (self.analyte0 * self.product_per_analyte)
            if self.side_product:
                Sd = float(X[i[self.side_product]])
                out["selectivity"] = P / (P + Sd) if P + Sd > 0 else 0.0
        out["ok"] = self.meets_values(out)
        return out

    def meets_values(self, v: dict, tol: float = 0.0) -> bool:
        if not np.isfinite(v.get("analyte", np.nan)):
            return False
        if self.goal in ("assay", "time"):
            return v["analyte"] <= self.target * (1 + tol)
        ok = v.get("yield", -1.0) >= self.yield_min * (1 - tol)
        if self.goal == "selectivity":
            ok = ok and v.get("selectivity", -1.0) >= self.selectivity_min * (1 - tol)
        return bool(ok)

    def meets(self, k, Et: float, rho: float, activity: dict | None = None) -> bool:
        if self.goal == "assay":  # the original test, kept identical (CADKIMM regression)
            return self.analyte_at(k, Et, rho, activity=activity) <= self.target
        return self.evaluate(k, Et, rho, activity)["ok"]

    def time_to_target(self, k, Et: float, rho: float, t_max: float, activity: dict | None = None) -> float:
        """First time the analyte is down to the target (linear interpolation on 2000 points); inf if never."""
        t = np.linspace(0.0, t_max, 2001)
        try:
            X = self.trajectory(k, Et, rho, t, activity)
        except SimulationError:
            return float("inf")
        y = X[:, self.network.index()[self.analyte]]
        hit = np.nonzero(y <= self.target)[0]
        if not len(hit):
            return float("inf")
        j = int(hit[0])
        if j == 0:
            return 0.0
        return float(t[j - 1] + (t[j] - t[j - 1]) * (y[j - 1] - self.target) / (y[j - 1] - y[j]))


def rho_min_equilibrium(conv: float, Kp: float) -> float:
    return conv + conv**2 / ((1 - conv) * Kp)


def equilibrium_state(overall: str, x0: dict, Kp: float) -> dict:
    """Small-molecule concentrations at the equilibrium of one overall reaction (enzyme-bound amounts neglected):
    the extent ξ solves ln Q(ξ) = ln K' (any stoichiometry; brentq on the feasible interval)."""
    from scipy.optimize import brentq

    st = Step.parse(overall)
    lhs, rhs = st.lhs, st.rhs
    x = {sp: float(x0.get(sp, 0.0)) for sp in list(lhs) + list(rhs)}
    hi = min(x[sp] / c for sp, c in lhs.items())
    lo = -min(x[sp] / c for sp, c in rhs.items())
    ln_k = float(np.log(Kp))

    def f(xi):
        num = sum(c * np.log(max(x[sp] + c * xi, 1e-300)) for sp, c in rhs.items())
        den = sum(c * np.log(max(x[sp] - c * xi, 1e-300)) for sp, c in lhs.items())
        return num - den - ln_k

    eps = 1e-12 * (hi - lo)
    if hi - lo <= 1e-300:
        xi = 0.0
    else:
        a_, b_ = lo + eps, hi - eps
        fa, fb = f(a_), f(b_)
        if fa >= 0:  # equilibrium at the product-free end (within 1e-12 of the range)
            xi = a_
        elif fb <= 0:  # complete reaction
            xi = b_
        else:
            xi = brentq(f, a_, b_, xtol=1e-16 + 1e-15 * (hi - lo), rtol=1e-14)
    out = dict(x)
    for sp, c in lhs.items():
        out[sp] = x[sp] - c * xi
    for sp, c in rhs.items():
        out[sp] = x[sp] + c * xi
    return out


def equilibrium_conversion_of(design: AssayDesign, rho: float) -> float:
    """Equilibrium conversion of the analyte at ratio ρ (``design.overall``; LDH formula without one)."""
    if not design.overall:
        return equilibrium_conversion(rho, design.Kp)
    eq = equilibrium_state(design.overall, design.initial(rho), design.Kp)
    return 1.0 - eq[design.analyte] / design.analyte0


def rho_min_of(design: AssayDesign) -> float | None:
    """Smallest ρ whose equilibrium allows ``conversion`` (None without a partner; inf if no ρ ≤ 10⁶ does)."""
    from scipy.optimize import brentq

    if not design.overall:
        return rho_min_equilibrium(design.conversion, design.Kp)
    if not design.partner:
        return None
    c = design.conversion

    def g(lr):
        return equilibrium_conversion_of(design, float(np.exp(lr))) - c

    lo, hi = np.log(1e-3), np.log(1e6)
    if g(hi) < 0:
        return float("inf")
    if g(lo) >= 0:
        return float(np.exp(lo))
    return float(np.exp(brentq(g, lo, hi, xtol=1e-12)))


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
    design: AssayDesign,
    rho: float,
    samples: list[np.ndarray],
    lo: float = 1e-6,
    hi: float = 1e-2,
    iters: int = 18,
    activities: list[dict | None] | None = None,
) -> float:
    """Smallest Et with analyte(time_s) ≤ (1 − conversion)·analyte0 for every sample (log bisection).
    ``activities``: enzyme activity a(t) per sample (inactivation), aligned with ``samples``."""
    target = design.target
    acts = activities or [None] * len(samples)

    def ok(Et):
        if design.goal == "assay":
            return all(design.analyte_at(k, Et, rho, activity=a) <= target for k, a in zip(samples, acts, strict=True))
        return all(design.meets(k, Et, rho, a) for k, a in zip(samples, acts, strict=True))

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
    design, rho, samples, lo, hi, iters, acts = args
    nominal = min_Et(design, rho, samples[:1], lo, hi, iters, acts[:1])
    return rho, nominal, min_Et(design, rho, samples, lo, hi, iters, acts)


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
    activities: list[dict | None] | None = None,
) -> DesignResult:
    """U-curve over ``rhos`` (nominal = first sample, robust = all samples), optimum, and a check of both designs
    on every sample. ``samples`` are rate vectors aligned with ``design.network.reactions()``; ``activities`` the
    enzyme activity a(t) of each sample (None = no inactivation)."""
    lo, hi = Et_bounds
    acts = list(activities) if activities is not None else [None] * len(samples)
    rmin = rho_min_of(design) if use_equilibrium_limit else None
    eq_rows = []
    if use_equilibrium_limit:
        if design.partner or not design.overall:
            for r in [1.0, 1.5, 2.0, 3.0, rmin, 8.0]:
                if r is None or not np.isfinite(r):
                    continue
                f = equilibrium_conversion_of(design, r)
                eq_rows.append({"rho": r, "conversion_eq": f, "NADH_left_uM": (1 - f) * design.analyte0 * 1e3})
        else:
            f = equilibrium_conversion_of(design, 1.0)
            eq_rows.append({"rho": None, "conversion_eq": f, "NADH_left_uM": (1 - f) * design.analyte0 * 1e3})
            if f < design.conversion:
                raise SimulationError(
                    f"평형 전환율 {f:.4f} < 목표 {design.conversion:g}: 효소량과 무관하게 도달할 수 없음(K' 확인)"
                )
    tasks = [(design, float(r), samples, lo, hi, iters, acts) for r in rhos]
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
            if design.goal == "assay":
                x = design.analyte_at(k, Et, rho_s, activity=acts[i]) if np.isfinite(Et) else float("nan")
                ev = {"analyte": x, "conversion": 1 - x / design.analyte0}
                fail = bool(not np.isfinite(x) or x > design.target * (1 + 1e-6))
            else:
                ev = design.evaluate(k, Et, rho_s, acts[i]) if np.isfinite(Et) else {"analyte": float("nan")}
                fail = not design.meets_values(ev, tol=1e-6)
            row = {
                "design": label,
                "sample": i,
                "Et_mM": float(Et),
                "NADH_at_target_mM": ev["analyte"],
                "conversion": ev.get("conversion", float("nan")),
                "fail": bool(fail),
            }
            for key in ("yield", "selectivity"):
                if key in ev:
                    row[key] = ev[key]
            check.append(row)
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


@dataclass
class TimeDesignResult:
    rhos: list[float]
    time_nominal: list[float]
    time_robust: list[float]
    rho_star: float
    Et: float
    time_star: float
    t_max: float

    def summary(self) -> dict:
        return {
            "goal": "time",
            "rho_star": self.rho_star,
            "Et_mM": self.Et,
            "Et_nM": self.Et * 1e6,
            "robust_time_s": self.time_star,
            "t_max_s": self.t_max,
        }


def _time_row(args):
    design, rho, samples, Et, t_max, acts = args
    ts = [design.time_to_target(k, Et, rho, t_max, a) for k, a in zip(samples, acts, strict=True)]
    return rho, ts[0], max(ts)


PEAK_WINDOW = 0.95  # stop-time window: worst-sample yield at least this fraction of its best


@dataclass
class PeakDesignResult:
    """Goal ``peak`` (a product that is made and then broken down again, e.g. kinetically controlled synthesis)."""

    rhos: list[float]
    rows: list[dict]  # per ρ: nominal peak, robust stop time, worst-sample yield, window, each sample's own peak
    rho_star: float
    Et: float
    t_max: float
    t: np.ndarray  # time grid at ρ*
    yields: np.ndarray  # (samples × time) yield at ρ*

    def star(self) -> dict:
        return next(r for r in self.rows if r["rho"] == self.rho_star)

    def summary(self) -> dict:
        s = self.star()
        return {
            "goal": "peak",
            "rho_star": self.rho_star,
            "Et_mM": self.Et,
            "Et_nM": self.Et * 1e6,
            "t_max_s": self.t_max,
            "robust_stop_s": s["stop_robust_s"],
            "robust_yield": s["yield_robust"],
            "stop_window_s": [s["window_lo_s"], s["window_hi_s"]],
            "nominal_peak_s": s["peak_nominal_s"],
            "nominal_peak_yield": s["yield_nominal"],
            "sample_peak_s": s["sample_peak_s"],
            "sample_peak_yield": s["sample_peak_yield"],
            "window_fraction": PEAK_WINDOW,
        }


def _peak_curves(args):
    design, rho, samples, Et, t_max, acts, n = args
    t = np.linspace(0.0, t_max, n)
    i = design.network.index()[design.product]
    scale = design.analyte0 * design.product_per_analyte
    Y = []
    for k, a in zip(samples, acts, strict=True):
        try:
            X = design.trajectory(k, Et, rho, t, a)
            Y.append(X[:, i] / scale)
        except SimulationError:
            Y.append(np.full(len(t), np.nan))
    return rho, t, np.array(Y)


def peak_row(rho: float, t: np.ndarray, Y: np.ndarray) -> dict:
    """Nominal peak (sample 0), robust stop time (maximises the worst sample's yield) and its window."""
    worst = np.nanmin(Y, axis=0)
    j = int(np.nanargmax(worst))
    ok = np.nonzero(worst >= PEAK_WINDOW * worst[j])[0]
    lo, hi = j, j  # the contiguous window around the best stop time
    while lo - 1 in ok:
        lo -= 1
    while hi + 1 in ok:
        hi += 1
    jn = int(np.nanargmax(Y[0]))
    own = [int(np.nanargmax(y)) for y in Y]
    return {
        "rho": float(rho),
        "peak_nominal_s": float(t[jn]),
        "yield_nominal": float(Y[0, jn]),
        "stop_robust_s": float(t[j]),
        "yield_robust": float(worst[j]),
        "window_lo_s": float(t[lo]),
        "window_hi_s": float(t[hi]),
        "sample_peak_s": [float(t[m]) for m in own],
        "sample_peak_yield": [float(Y[s, m]) for s, m in enumerate(own)],
        "yield_at_stop": [float(y[j]) for y in Y],
    }


def run_peak_design(
    design: AssayDesign,
    samples: list[np.ndarray],
    rhos: list[float],
    Et: float,
    t_max: float,
    n_points: int = 1201,
    workers: int | None = None,
    activities: list[dict | None] | None = None,
) -> PeakDesignResult:
    """Goal ``peak``: with the enzyme amount ``Et`` fixed, per ρ the stop time that maximises the worst sample's
    yield of ``design.product``; ρ* has the highest worst-sample yield. Nominal = sample 0's own peak."""
    if not design.product:
        raise ValueError("design.target peak: design.product(만들려는 생성물)가 필요함")
    acts = list(activities) if activities is not None else [None] * len(samples)
    tasks = [(design, float(r), samples, Et, t_max, acts, n_points) for r in rhos]
    w = workers if workers is not None else n_workers(len(tasks))
    if w > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=w) as ex:
            curves = list(ex.map(_peak_curves, tasks))
    else:
        curves = [_peak_curves(t) for t in tasks]
    rows = [peak_row(r, t, Y) for r, t, Y in curves]
    j = int(np.argmax([r["yield_robust"] for r in rows]))
    _r, t, Y = curves[j]
    return PeakDesignResult([float(r) for r in rhos], rows, rows[j]["rho"], float(Et), float(t_max), t, Y)


def run_time_design(
    design: AssayDesign,
    samples: list[np.ndarray],
    rhos: list[float],
    Et: float,
    t_max: float,
    workers: int | None = None,
    activities: list[dict | None] | None = None,
) -> TimeDesignResult:
    """Goal ``time``: with the enzyme amount ``Et`` fixed, the ρ whose worst sample reaches the conversion soonest."""
    acts = list(activities) if activities is not None else [None] * len(samples)
    tasks = [(design, float(r), samples, Et, t_max, acts) for r in rhos]
    w = workers if workers is not None else n_workers(len(tasks))
    if w > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=w) as ex:
            rows = list(ex.map(_time_row, tasks))
    else:
        rows = [_time_row(t) for t in tasks]
    rob = [r[2] for r in rows]
    if not np.isfinite(min(rob)):
        raise SimulationError(f"효소량 {Et * 1e6:.4g} nM로는 어떤 ρ에서도 {t_max:g} s 안에 목표 전환율에 닿지 않음")
    j = int(np.argmin(rob))
    return TimeDesignResult(
        [r[0] for r in rows], [r[1] for r in rows], rob, float(rows[j][0]), float(Et), float(rob[j]), float(t_max)
    )
